/*
 * Route Story offline scene page. Plain ES2020, no build step: this file is
 * served verbatim by BundleServer under /scene/scene.js next to the vendored
 * MapLibre build. Every relative asset URL below is written against the page
 * (/scene/index.html), not against this script's own location, since browser
 * fetch/relative-URL resolution is always page-relative.
 *
 * window.routeStory.init(config) builds the style and starts loading tiles.
 * window.routeStory.ready resolves once the style and label data are loaded.
 * window.routeStory.renderPose(pose) jumps the camera, updates the marker and
 * light, waits for the frame to be fully settled, and resolves with the
 * projected label positions for that pose. An optional pose.route (a list of
 * coordinate lists) supplies the dim whole-route line and an optional
 * pose.travelled (same shape) the bright travelled line -- both geometries
 * come from the caller, never derived here, so the travelled line can never
 * disagree with where the marker actually is. Both are sent on EVERY pose,
 * because a pose is what a frame cache keys a rendered frame on and must
 * therefore fully determine the pixels on its own -- a "send once" that
 * relied on this page having already seen an earlier pose would leave a
 * fresh page (or one that reached a cache hit first) with no route line at
 * all. Both are drawn by drawRouteOverlay straight off the pose in screen
 * space (see its own comment) -- renderPose no longer feeds them into the
 * "route"/"travelled" MapLibre sources, whose only layers are
 * visibility:"none" and painted nothing.
 * An optional pose.marker_heading rotates a chevron glyph on the marker to
 * face the direction of travel. An optional pose.pins (a list of [lon, lat]
 * pairs, already filtered to "already played" by the caller) draws a small
 * pin at each one, under the marker and over the route line.
 */
(function () {
  "use strict";

  function tileURL(kind, extension) {
    return "../" + kind + "/{z}/{x}/{y}." + extension;
  }

  var EMPTY_FEATURE_COLLECTION = { type: "FeatureCollection", features: [] };

  // The route's dim (not-yet-travelled) and bright (travelled) colours.
  // ROUTE_BRIGHT_COLOR matches the pre-overlay "travelled-line" layer's
  // static "line-color" exactly, so the look is unchanged from before.
  var ROUTE_BRIGHT_COLOR = "#ffcf3f";
  var ROUTE_DIM_COLOR = "#5c4c1f";

  // MapLibre drapes a line layer onto 3D terrain by extruding its width in
  // the map's own 2D plane, not in screen space -- so at a grazing view
  // angle (a steep pitch looking nearly along the road, which is common on
  // steep/switchback terrain) the extruded offset itself foreshortens on
  // screen and no paint "line-width" value fixes it (even a wide
  // paint width can still land sub-pixel near the marker). The ENTIRE
  // on-screen route and travelled line are instead drawn on a plain 2D
  // overlay canvas, in true screen pixels, on top of the (now-hidden)
  // draped "route-casing"/"route-line"/"travelled-line" layers -- see
  // drawRouteOverlay. A radius-limited redraw (an earlier version of this
  // fix) left a visible seam where the redraw stopped and the draped
  // hairline resumed; drawing the whole visible line removes the seam by
  // construction, and the two hidden draped layers are kept in the style
  // (never removed) purely so their "line-width"/"line-color" paint stays
  // documented and easy to restore if the overlay is ever reverted.
  var ROUTE_LINE_COLOR = "rgba(255, 255, 255, 0.55)";
  var ROUTE_HALO_COLOR = "rgba(10, 26, 42, 0.75)";
  // Every route stroke is a fraction of the frame's width, exactly as
  // route_story/layers.py computes it (route_halo_width_px / route_line_width_px /
  // travelled_line_width_px, divisors 220 / 500 / 280). They were flat numbers
  // here while the pins below already scaled with frame height, so a 1440p
  // master grew its pins and kept its 1080p line -- a quarter thinner against
  // everything around it. tests/test_route_story_line_width_parity.py reads
  // the divisors out of Python and asserts these three expressions verbatim,
  // so the renderers cannot drift apart silently.
  function routeHaloWidthPx(width) {
    return Math.max(5, Math.floor(width / 220));
  }
  function routeLineWidthPx(width) {
    return Math.max(2, Math.floor(width / 500));
  }
  function travelledLineWidthPx(width) {
    return Math.max(4, Math.floor(width / 280));
  }
  // A small pin marking where an already-played photo/video beat sits on the
  // route -- white with a dark outline, deliberately far in colour from both
  // ROUTE_BRIGHT_COLOR and the marker fill so neither frame_qa's travelled-
  // line detector nor a viewer confuses the two. Radius scales with frame
  // height (~6px at 1080p, route_story/layers.py's `_draw_map_pins` mirrors this).
  var PIN_FILL_COLOR = "#ffffff";
  var PIN_OUTLINE_COLOR = "#071219";
  var PIN_OUTLINE_WIDTH_PX = 2;
  // The day's start and finish, pinned on every frame from pose.endpoints.
  // Mirrors route_story/layers.py's ROUTE_START_RGB / ROUTE_FINISH_RGB exactly --
  // the two renderers must agree, or a film that switches renderers mid-story
  // changes the colour of the same place. Both are far in colour from
  // ROUTE_BRIGHT_COLOR, the marker fill and PIN_FILL_COLOR.
  var ROUTE_START_COLOR = "#35d07a";
  var ROUTE_FINISH_COLOR = "#e24c6f";
  // A vertex is only drawn if it is within this margin of the viewport --
  // generous enough that a line entering the frame at an angle is never cut
  // short of the actual edge, small enough not to waste time subdividing
  // geometry that is nowhere near visible.
  var VIEWPORT_MARGIN_PX = 60;
  // Long spans between two consecutive route vertices are resampled every
  // this many screen px before the occlusion test runs, so a segment that
  // dips behind a ridge for only part of its length is cut exactly at the
  // ridge instead of being kept or dropped whole.
  var OCCLUSION_SUBDIVIDE_PX = 12;
  // Hard cap on subdivision steps for a single source-to-source span. A
  // route point behind the camera (or otherwise degenerate for this pose's
  // projection) can project to a wild or non-finite screen coordinate, which
  // would otherwise make spanPx huge and the per-vertex subdivide loop below
  // run for an effectively unbounded number of iterations on the page's main
  // thread -- blocking "idle" from ever firing and hanging renderPose. Every
  // span the whole route ships is short in the real world (consecutive GPX
  // points), so a genuinely visible span never approaches this cap; it only
  // ever bites a degenerate projection.
  var MAX_SUBDIVIDE_STEPS = 256;
  // isVertexOccluded's tolerance, in on-screen px worth of ground distance
  // at the vertex's own latitude/zoom (see groundResolutionMetersPerPixel).
  // A visible vertex's project->unproject round trip recovers the
  // same lng/lat to float noise, while a vertex genuinely
  // hidden behind a ridge recovers a DIFFERENT point, tens to hundreds of metres away (the
  // ridge surface MapLibre's terrain-aware unproject actually raycasts to
  // at that pixel -- see isVertexOccluded). 4px of ground resolution is
  // comfortably above the float-noise floor and comfortably below any
  // genuine occlusion case, including a narrow ridge.
  var OCCLUSION_TOLERANCE_PX = 4;
  // How far below sea level labelCullRadiusM assumes the ground can lie. The
  // lowest dry land on Earth is ~430 m down; anything lower than this would
  // only ever make the cull too generous about a label that cannot exist.
  var LABEL_CULL_FLOOR_M = 500;
  // Largest raycast miss, in metres, a hidden run between visible samples may
  // have and still be DEM noise (see repairShallowOcclusionRuns). DEM noise
  // stays well below this; a genuinely hidden vertex misses by more.
  var SHALLOW_OCCLUSION_MAX_M = 30;

  function buildStyle(styleConfig) {
    return {
      version: 8,
      sources: {
        imagery: {
          type: "raster",
          tiles: [tileURL("imagery", "jpg")],
          tileSize: 256,
          minzoom: styleConfig.imagery.minzoom,
          maxzoom: styleConfig.imagery.maxzoom,
        },
        terrainSource: {
          type: "raster-dem",
          tiles: [tileURL("terrain", "png")],
          tileSize: 256,
          encoding: "mapbox",
          minzoom: styleConfig.terrain.minzoom,
          maxzoom: styleConfig.terrain.maxzoom,
        },
        route: { type: "geojson", data: EMPTY_FEATURE_COLLECTION },
        travelled: { type: "geojson", data: EMPTY_FEATURE_COLLECTION },
        marker: { type: "geojson", data: { type: "FeatureCollection", features: [] } },
      },
      terrain: { source: "terrainSource", exaggeration: styleConfig.exaggeration || 1 },
      // Subtle sky only; never invents clouds or a sun disc.
      sky: {
        "sky-color": "#88b6e0",
        "horizon-color": "#e6eef5",
        "fog-color": "#e6eef5",
        "sky-horizon-blend": 0.5,
        "horizon-fog-blend": 0.3,
        "fog-ground-blend": 0.2,
      },
      // A dark neutral horizon haze in place of page-white beyond bundle
      // coverage, and to hide the DEM skirt at the terrain tile edge.
      fog: {
        range: [0.5, 10],
        color: "#101418",
        "horizon-blend": 0.2,
      },
      layers: [
        { id: "background", type: "background", paint: { "background-color": "#101418" } },
        { id: "imagery", type: "raster", source: "imagery" },
        {
          id: "hillshade",
          type: "hillshade",
          source: "terrainSource",
          paint: { "hillshade-exaggeration": 0.6 },
        },
        // These three draped layers are never rendered (see
        // drawRouteOverlay's own screen-space redraw of the whole visible
        // route/travelled line, on the #overlay canvas). Kept in the style
        // -- not removed -- purely to keep their paint documented; "route"/
        // "travelled" sources are still fed every pose (setRoute/
        // setTravelled) since drawRouteOverlay reads the SAME pose fields,
        // never a second copy of the geometry.
        {
          id: "route-casing",
          type: "line",
          source: "route",
          layout: { visibility: "none" },
          paint: { "line-color": "#0a1a2a", "line-width": 6, "line-opacity": 0.85 },
        },
        {
          id: "route-line",
          type: "line",
          source: "route",
          layout: { visibility: "none" },
          paint: { "line-color": ROUTE_DIM_COLOR, "line-width": 3 },
        },
        {
          id: "travelled-line",
          type: "line",
          source: "travelled",
          layout: { visibility: "none" },
          paint: { "line-color": ROUTE_BRIGHT_COLOR, "line-width": 3 },
        },
        {
          id: "marker",
          type: "circle",
          source: "marker",
          paint: {
            "circle-radius": 7,
            "circle-color": "#ff3b30",
            "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 2,
          },
        },
        // Only draws when the marker feature carries a "heading" property
        // (see setMarker) -- a pose with no marker_heading leaves this layer
        // with nothing to render, i.e. the plain circle above, as today.
        {
          id: "marker-heading",
          type: "symbol",
          source: "marker",
          filter: ["has", "heading"],
          layout: {
            "icon-image": MARKER_HEADING_ICON,
            "icon-rotate": ["get", "heading"],
            "icon-rotation-alignment": "map",
            "icon-allow-overlap": true,
            "icon-ignore-placement": true,
            "icon-size": 0.6,
          },
        },
      ],
    };
  }

  // Drawn in-page on an offscreen canvas -- never fetched, so the scene page
  // stays fully offline (see is_allowed_request in capture.py).
  var MARKER_HEADING_ICON = "route-story-chevron";

  function buildHeadingChevron() {
    var size = 32;
    var canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(size / 2, 2);
    ctx.lineTo(size - 4, size - 6);
    ctx.lineTo(size / 2, size - 14);
    ctx.lineTo(4, size - 6);
    ctx.closePath();
    ctx.fill();
    return ctx.getImageData(0, 0, size, size);
  }

  // The overlay canvas is sized to the requested frame once, in init() --
  // it never changes size again, since a capture's width/height is fixed
  // for the whole scene page's lifetime (one page per SceneCapture).
  function sizeOverlay(width, height) {
    var canvas = document.getElementById("overlay");
    canvas.width = width;
    canvas.height = height;
  }

  // Approximate great-circle distance in metres -- good enough at the
  // scale (metres to low km) this is used at.
  function haversineMeters(a, b) {
    var R = 6371000;
    var lat1 = (a[1] * Math.PI) / 180;
    var lat2 = (b[1] * Math.PI) / 180;
    var dLat = lat2 - lat1;
    var dLon = ((b[0] - a[0]) * Math.PI) / 180;
    var h =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  // Standard Web Mercator ground resolution (metres per screen px) at a
  // given zoom and latitude.
  function groundResolutionMetersPerPixel(zoom, lat) {
    return (156543.03392 * Math.cos((lat * Math.PI) / 180)) / Math.pow(2, zoom);
  }

  // Whether ``lngLat`` is hidden behind terrain from the current camera.
  // MapLibre's terrain-aware unproject raycasts the actual visible surface
  // at a screen point (see OCCLUSION_TOLERANCE_PX's comment): projecting a VISIBLE
  // vertex and unprojecting the result recovers the same lng/lat to float
  // precision, while a vertex a nearer ridge hides recovers the ridge's own
  // (very different) lng/lat instead, because that is what is actually
  // drawn at that pixel.
  function isVertexOccluded(lngLat, zoom) {
    return occlusionMissMeters(lngLat) > occlusionToleranceMeters(lngLat, zoom);
  }

  // How far, in metres, the project->unproject round trip lands from
  // ``lngLat``: ~0 for a visible vertex, the hiding ridge's distance for a
  // hidden one (see isVertexOccluded).
  function occlusionMissMeters(lngLat) {
    var screen = map.project(lngLat);
    var recovered = map.unproject([screen.x, screen.y]);
    return haversineMeters(lngLat, [recovered.lng, recovered.lat]);
  }

  function occlusionToleranceMeters(lngLat, zoom) {
    return OCCLUSION_TOLERANCE_PX * groundResolutionMetersPerPixel(zoom, lngLat[1]);
  }

  function withinViewport(point, width, height) {
    return (
      point.x >= -VIEWPORT_MARGIN_PX &&
      point.x <= width + VIEWPORT_MARGIN_PX &&
      point.y >= -VIEWPORT_MARGIN_PX &&
      point.y <= height + VIEWPORT_MARGIN_PX
    );
  }

  // Projects one vertex and records, SEPARATELY, whether it is off-canvas and
  // whether terrain hides it. They are kept apart because only the occlusion
  // verdict is repairable (see repairIsolatedOcclusions) -- an off-canvas
  // vertex has no pixels to draw whatever its neighbours say.
  function vertexSample(lngLat, zoom, width, height) {
    var point = map.project(lngLat);
    var missM = occlusionMissMeters(lngLat);
    return {
      x: point.x,
      y: point.y,
      onCanvas: withinViewport(point, width, height),
      occluded: missM > occlusionToleranceMeters(lngLat, zoom),
      missM: missM,
    };
  }

  // A SINGLE hidden vertex between two visible ones is a raycast artifact,
  // not a ridge, and this repairs it in place.
  //
  // isVertexOccluded asks MapLibre to unproject the vertex's own screen point
  // and calls the vertex hidden when the raycast lands somewhere else on the
  // ground. Near a terrain tile edge -- or wherever the DEM sample the ray
  // happens to hit belongs to the next post over -- that round trip can come
  // back a few metres off for one vertex while both its neighbours recover to
  // float precision. When that happens the vertex can miss the recovery
  // tolerance by a small margin, with visible vertices on both sides and the
  // marker itself visible. That one false verdict breaks the subpath short of
  // the marker, so the drawn line stops short of a dot that is always
  // drawn: exactly the "red dot lags behind the bright line" defect
  // frame_qa's max_line_end_to_marker_px exists to catch.
  //
  // Repairing a run of ONE cannot draw a line through a mountain: a real ridge
  // hides a contiguous run of vertices (genuinely hidden vertices recover far
  // deeper misses, and strokeRouteParts subdivides every span to
  // OCCLUSION_SUBDIVIDE_PX so a real crossing always produces several in a
  // row), and any run of two or more is left hidden.
  function repairIsolatedOcclusions(samples) {
    for (var i = 1; i < samples.length - 1; i++) {
      if (samples[i].occluded && !samples[i - 1].occluded && !samples[i + 1].occluded) {
        samples[i].occluded = false;
      }
    }
    return samples;
  }

  // A hidden RUN between two visible samples whose every raycast missed by
  // less than SHALLOW_OCCLUSION_MAX_M is DEM noise too, however long it is.
  // Typical case: the marker recovers exactly, yet a couple of samples a few
  // pixels before it miss by more than the recovery tolerance, with a visible
  // sample between them and the marker.
  // Too long for repairIsolatedOcclusions, not adjacent to the marker for
  // forceTrailingSpanToMarker, so the bright line broke right before the dot.
  // A genuinely hidden sample recovers the hiding ridge far deeper than
  // SHALLOW_OCCLUSION_MAX_M, so one deep sample keeps its whole run hidden; a
  // sample with no measured miss is never called shallow.
  function repairShallowOcclusionRuns(samples) {
    var i = 1;
    while (i < samples.length - 1) {
      if (!samples[i].occluded || samples[i - 1].occluded) {
        i++;
        continue;
      }
      var end = i;
      var shallow = true;
      while (end < samples.length && samples[end].occluded) {
        if (!(samples[end].missM < SHALLOW_OCCLUSION_MAX_M)) {
          shallow = false;
        }
        end++;
      }
      if (shallow && end < samples.length) {
        for (var k = i; k < end; k++) {
          samples[k].occluded = false;
        }
      }
      i = end + 1;
    }
    return samples;
  }

  // The travelled line's own last vertex is always the marker's own
  // coordinate (see _travelled_coordinates in scene_renderer.py), so when
  // the marker itself is visible the ground right under it is, by
  // construction, not occluded -- the failure here is a RUN of
  // two-plus falsely-occluded vertices adjacent to a visible marker, too
  // long for repairIsolatedOcclusions (which only repairs a lone vertex) to
  // touch. This forces just that trailing run -- walking back from the
  // vertex before the marker until it hits one already visible -- so the
  // drawn line always reaches a marker that is always drawn. A hidden run
  // further back along the path, not adjacent to the marker, is left alone:
  // it is a real ridge. An invisible marker forces nothing, so a hidden
  // marker never gets a fabricated line drawn to it.
  function forceTrailingSpanToMarker(samples, markerVisible) {
    if (!markerVisible) {
      return samples;
    }
    for (var i = samples.length - 2; i >= 0 && samples[i].occluded; i--) {
      samples[i].occluded = false;
    }
    return samples;
  }

  // Strokes every part of a MultiLineString-shaped coordinate list in
  // constant screen pixels. Long spans between two consecutive source
  // vertices are resampled (by linear lng/lat interpolation) every
  // OCCLUSION_SUBDIVIDE_PX of screen distance, so a span that goes behind a
  // ridge partway along is cut exactly there rather than kept or dropped
  // whole.
  function strokeRouteParts(ctx, parts, zoom, width, height, markerVisible) {
    (parts || []).forEach(function (part, partIndex) {
      if (!part || part.length < 2) {
        return;
      }
      // Every vertex is sampled BEFORE anything is drawn: the isolated-
      // occlusion repair needs both neighbours of a vertex, which a streaming
      // moveTo/lineTo pass cannot see.
      var samples = [vertexSample(part[0], zoom, width, height)];
      for (var i = 1; i < part.length; i++) {
        var prev = part[i - 1];
        var curr = part[i];
        var prevScreen = map.project(prev);
        var currScreen = map.project(curr);
        var spanPx = Math.hypot(currScreen.x - prevScreen.x, currScreen.y - prevScreen.y);
        var steps = Number.isFinite(spanPx)
          ? Math.min(MAX_SUBDIVIDE_STEPS, Math.max(1, Math.ceil(spanPx / OCCLUSION_SUBDIVIDE_PX)))
          : 1;
        for (var step = 1; step <= steps; step++) {
          var t = step / steps;
          var lngLat = [prev[0] + (curr[0] - prev[0]) * t, prev[1] + (curr[1] - prev[1]) * t];
          samples.push(vertexSample(lngLat, zoom, width, height));
        }
      }
      repairIsolatedOcclusions(samples);
      repairShallowOcclusionRuns(samples);
      if (partIndex === parts.length - 1) {
        forceTrailingSpanToMarker(samples, markerVisible);
      }
      ctx.beginPath();
      var drawing = false;
      for (var s = 0; s < samples.length; s++) {
        var sample = samples[s];
        if (!sample.onCanvas || sample.occluded) {
          drawing = false;
          continue;
        }
        if (drawing) {
          ctx.lineTo(sample.x, sample.y);
        } else {
          ctx.moveTo(sample.x, sample.y);
        }
        drawing = true;
      }
      ctx.stroke();
    });
  }

  // Redraws the WHOLE on-screen route (dim, under) and travelled line
  // (bright, over) in constant screen pixels, then knocks the marker's own
  // disc back out of whatever line pixels landed under it. ``pose`` supplies
  // every input (route/travelled/marker/zoom) -- all already part of the
  // pose the frame cache keys on, so this stays fully determined by it.
  function drawRouteOverlay(pose) {
    var canvas = document.getElementById("overlay");
    var ctx = canvas.getContext("2d");
    var width = canvas.width;
    var height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    var zoom = pose.zoom;
    var markerVisible = Boolean(pose.marker) && !isVertexOccluded(pose.marker, zoom);

    if (pose.route) {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = ROUTE_HALO_COLOR;
      ctx.lineWidth = routeHaloWidthPx(width);
      strokeRouteParts(ctx, pose.route, zoom, width, height);
      ctx.strokeStyle = ROUTE_LINE_COLOR;
      ctx.lineWidth = routeLineWidthPx(width);
      strokeRouteParts(ctx, pose.route, zoom, width, height);
    }
    if (pose.travelled) {
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.strokeStyle = ROUTE_BRIGHT_COLOR;
      ctx.lineWidth = travelledLineWidthPx(width);
      strokeRouteParts(ctx, pose.travelled, zoom, width, height, markerVisible);
    }
    if (pose.pins && pose.pins.length) {
      var pinRadius = Math.max(3, Math.round((height * 6) / 1080));
      ctx.fillStyle = PIN_FILL_COLOR;
      ctx.strokeStyle = PIN_OUTLINE_COLOR;
      ctx.lineWidth = PIN_OUTLINE_WIDTH_PX;
      pose.pins.forEach(function (lngLat) {
        var point = map.project(lngLat);
        if (!withinViewport(point, width, height)) {
          return;
        }
        ctx.beginPath();
        ctx.arc(point.x, point.y, pinRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }
    if (pose.endpoints && pose.endpoints.length) {
      // Drawn after the photo pins and before the marker: an endpoint
      // outranks a photo pin sitting on it, the marker outranks everything.
      var endpointRadius = Math.max(4, Math.round((height * 8) / 1080));
      ctx.strokeStyle = PIN_OUTLINE_COLOR;
      ctx.lineWidth = PIN_OUTLINE_WIDTH_PX;
      pose.endpoints.forEach(function (lngLat, index) {
        var point = map.project(lngLat);
        if (!withinViewport(point, width, height)) {
          return;
        }
        ctx.fillStyle = index === 0 ? ROUTE_START_COLOR : ROUTE_FINISH_COLOR;
        ctx.beginPath();
        ctx.arc(point.x, point.y, endpointRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    }
    if (pose.marker) {
      // The marker circle layer's own radius/stroke -- read, never
      // guessed, so a future style change to the marker can't silently
      // leave a yellow ring around it again.
      var radius = map.getPaintProperty("marker", "circle-radius") || 0;
      var strokeWidth = map.getPaintProperty("marker", "circle-stroke-width") || 0;
      var markerPoint = map.project(pose.marker);
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.arc(markerPoint.x, markerPoint.y, radius + strokeWidth + 1, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  var map = null;
  // The last route geometry actually applied via setData, as its
  // JSON.stringify text -- lets setRoute skip a re-tile when the pose sends
  // the same (usually unchanged, whole-trip) route again, which it now does
  // on every pose (see renderPose).
  var lastRouteJSON = null;
  var labels = [];
  var tileFailures = [];
  // How often renderPose's watchdog checks for tile failures and the
  // deadline (see renderPose) -- it never decides success.
  var SETTLE_POLL_MS = 60;
  var readyResolve;
  // Polled by the capture under a deadline it enforces itself: a promise the
  // driver awaits has no timeout, and "load" is not guaranteed to fire.
  var isReady = false;
  // Readiness is the AND of two independent waits (see init). When it never
  // arrives, "not ready" alone cannot say which half is outstanding, and the
  // driver has no way to reach into this closure to find out -- so these two
  // flags, and diagnose() below, are the scene's own account of the stall.
  var loadFired = false;
  var labelsLoaded = false;
  var ready = new Promise(function (resolve) {
    readyResolve = resolve;
  });

  function loadLabels() {
    return fetch("../labels.geojson")
      .then(function (resp) {
        return resp.ok ? resp.json() : { features: [] };
      })
      .then(function (geojson) {
        labels = geojson.features || [];
      })
      .catch(function () {
        labels = [];
      });
  }

  function init(config) {
    var mapOptions = {
      container: "map",
      style: buildStyle(config.style),
      center: [0, 0],
      zoom: 0,
      interactive: false,
      attributionControl: false,
      fadeDuration: 0,
      pixelRatio: 1,
      maxTileCacheSize: 4096,
      canvasContextAttributes: { preserveDrawingBuffer: true },
    };
    map = new maplibregl.Map(mapOptions);
    sizeOverlay(config.width, config.height);
    map.on("error", recordTileFailure);
    // Added synchronously, well before "load" -- the "marker-heading" layer
    // above references it from the very first style load, so frame 0 must
    // never race an image that isn't there yet.
    map.addImage(MARKER_HEADING_ICON, buildHeadingChevron());
    Promise.all([
      new Promise(function (resolve) {
        map.once("load", function () {
          loadFired = true;
          resolve();
        });
      }),
      loadLabels().then(function () {
        labelsLoaded = true;
      }),
    ]).then(function () {
      isReady = true;
      readyResolve();
    });
  }

  function setLight(light) {
    if (!light || !map.getLayer("hillshade")) {
      return;
    }
    // Hillshade only exposes an illumination *direction*; altitude has no
    // direct paint-property equivalent and is intentionally not simulated.
    map.setPaintProperty("hillshade", "hillshade-illumination-direction", light.azimuth);
    // A light carrying only `azimuth` (the pre-sun-model shape) stops here:
    // direction-only, exactly as before. `elevation`/`intensity` arrive
    // together from SceneLight.to_scene(), never one without the other.
    if (light.elevation !== undefined && light.intensity !== undefined) {
      // MapLibre's light.position is [radial, azimuthal deg, polar deg from
      // zenith]; anchor "map" makes the azimuthal term a compass bearing.
      // Night intensity is already low from the Python side -- nothing here
      // invents moonlight.
      map.setLight({
        anchor: "map",
        position: [1.15, light.azimuth, 90 - light.elevation],
        intensity: light.intensity,
      });
    }
    if (light.hillshadeExaggeration !== undefined) {
      map.setPaintProperty("hillshade", "hillshade-exaggeration", light.hillshadeExaggeration);
    }
  }

  function setMarker(marker, heading) {
    var source = map.getSource("marker");
    if (!source) {
      return;
    }
    var features = [];
    if (marker) {
      var properties = {};
      if (heading !== null && heading !== undefined) {
        properties.heading = heading;
      }
      features.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: marker },
        properties: properties,
      });
    }
    source.setData({ type: "FeatureCollection", features: features });
  }

  // ``coordinates`` is a list of coordinate lists (MultiLineString shape),
  // already cut at every gap by the caller -- a part shorter than 2 points
  // is dropped rather than passed to MapLibre as an invalid LineString.
  function multiLineFeatureCollection(coordinates) {
    var features = (coordinates || [])
      .filter(function (part) {
        return part && part.length >= 2;
      })
      .map(function (part) {
        return {
          type: "Feature",
          geometry: { type: "LineString", coordinates: part },
          properties: {},
        };
      });
    return { type: "FeatureCollection", features: features };
  }

  function setRoute(coordinates) {
    var source = map.getSource("route");
    if (!source) {
      return;
    }
    // Sent on every pose now (a pose must fully describe its own frame --
    // it is the frame cache's key), but the geometry itself is normally the
    // whole, unchanged trip: re-tiling the same source every frame would
    // needlessly cost a repaint and delay "idle" for no visual change.
    var json = JSON.stringify(coordinates);
    if (json === lastRouteJSON) {
      return;
    }
    lastRouteJSON = json;
    source.setData(multiLineFeatureCollection(coordinates));
  }

  function setTravelled(coordinates) {
    var source = map.getSource("travelled");
    if (!source) {
      return;
    }
    source.setData(multiLineFeatureCollection(coordinates));
  }

  // How far from the view centre (metres) a label can sit and still land on
  // canvas -- a bound from the camera's own geometry, never from
  // map.getBounds(): with terrain on, the bounds come from where the corner
  // rays hit the mesh, so a near ridge shrinks them and far-field labels the
  // frame plainly shows fall outside (for example it can drop a named gap
  // at a steep pitch). The farthest ground a frame can show is where its top-corner
  // ray lands, and that ray travels farthest over the LOWEST ground, so the
  // drop is taken to `floorM` below sea level. Infinity means "do not cull":
  // the horizon is in frame, or the view is wide enough that one latitude's
  // metres-per-pixel no longer describes it.
  function labelCullRadiusM(view) {
    var toRad = Math.PI / 180;
    var pitch = view.pitchDeg * toRad;
    var t = Math.tan((view.fovDeg * toRad) / 2);
    var aspect = view.width / view.height;
    var descent = Math.cos(pitch) - t * Math.sin(pitch);
    if (!(descent > 0.02)) {
      return Infinity;
    }
    var metresPerPixel =
      (40075016.686 * Math.cos(view.latDeg * toRad)) / (512 * Math.pow(2, view.zoom));
    var cameraToCentreM = (view.height / 2 / t) * metresPerPixel;
    var dropM = cameraToCentreM * Math.cos(pitch) + view.centreElevationM + view.floorM;
    var reach = Math.sqrt(Math.pow(aspect * t, 2) + Math.pow(t * Math.cos(pitch) + Math.sin(pitch), 2));
    // From the point under the camera; the centre sits ahead of it, so add
    // that offset (triangle inequality) and a tenth for the flat-earth maths.
    var radius = ((dropM * reach) / descent + cameraToCentreM * Math.sin(pitch)) * 1.1;
    return radius > 300000 ? Infinity : radius;
  }

  function cullLabelsToRadius(features, centreLng, centreLat, radiusM) {
    if (!isFinite(radiusM)) {
      return features;
    }
    var metresPerDegLat = 111320;
    var metresPerDegLng = 111320 * Math.cos((centreLat * Math.PI) / 180);
    var limit = radiusM * radiusM;
    return features.filter(function (feature) {
      var coords = feature.geometry.coordinates;
      var dx = (coords[0] - centreLng) * metresPerDegLng;
      var dy = (coords[1] - centreLat) * metresPerDegLat;
      return dx * dx + dy * dy <= limit;
    });
  }

  // The marker rides along as one more entry, kind "marker" with no name:
  // the pill layout (proof.place_labels) keeps every pill clear of it, and it
  // needs the marker where THIS pose's camera and terrain actually put it.
  function projectLabels(marker) {
    var canvas = map.getCanvas();
    var width = canvas.width;
    var height = canvas.height;
    function onCanvas(point) {
      return point.x >= 0 && point.x <= width && point.y >= 0 && point.y <= height;
    }
    // Only what can actually be drawn crosses back to Python. labels.geojson
    // holds thousands of features for a long trip, and every consumer there already
    // requires `visible` (proof.place_labels skips a label without it; both
    // marker lookups test it), so off-canvas entries would be projected,
    // serialised, shipped and then discarded -- a large cost across a
    // long film, for a handful of labels a frame. The filter
    // runs BEFORE the marker is pushed: the marker is not a label and rides
    // back whatever its own flag says.
    //
    // And only what could be on canvas is PROJECTED. map.project is
    // terrain-aware (an elevation lookup per point), so projecting every
    // feature would dominate each pose on the main thread, for the handful
    // drawn. A plain distance test first leaves a few hundred at most to project.
    var centre = map.getCenter();
    var canvasSize = map.getCanvas();
    var projected = cullLabelsToRadius(
      labels,
      centre.lng,
      centre.lat,
      labelCullRadiusM({
        zoom: map.getZoom(),
        pitchDeg: map.getPitch(),
        latDeg: centre.lat,
        fovDeg: map.getVerticalFieldOfView(),
        width: canvasSize.width,
        height: canvasSize.height,
        centreElevationM: Math.max(0, map.getCameraTargetElevation() || 0),
        floorM: LABEL_CULL_FLOOR_M,
      })
    )
      .map(function (feature) {
        var coords = feature.geometry.coordinates;
        var point = map.project(coords);
        return {
          name: feature.properties.name,
          kind: feature.properties.kind,
          x: point.x,
          y: point.y,
          visible: onCanvas(point),
        };
      })
      .filter(function (entry) {
        return entry.visible;
      });
    if (marker) {
      var at = map.project(marker);
      projected.push({ name: "", kind: "marker", x: at.x, y: at.y, visible: onCanvas(at) });
    }
    return projected;
  }

  // A tile the bundle does not hold answers 404; MapLibre then marks that
  // tile errored, areTilesLoaded() never turns true again, and a wait for
  // "settled" would hang for the rest of the film. Remember the failures so
  // renderPose can fail with the URL that caused them.
  function recordTileFailure(event) {
    var url = (event && event.sourceURL) || (event && event.error && event.error.url) || "";
    var message = (event && event.error && event.error.message) || "unknown error";
    tileFailures.push(url ? "tile " + url + " failed: " + message : message);
  }

  function renderPose(pose, settleTimeoutMs) {
    var budget = settleTimeoutMs > 0 ? settleTimeoutMs : 20000;
    tileFailures.length = 0;
    map.jumpTo({
      center: [pose.longitude, pose.latitude],
      zoom: pose.zoom,
      pitch: pose.pitch || 0,
      bearing: pose.bearing || 0,
    });
    setMarker(pose.marker || null, pose.marker_heading);
    setLight(pose.light || null);
    // setRoute/setTravelled (still defined below, unused) used to be called
    // here every pose. They feed the "route"/"travelled" GeoJSON sources,
    // whose only consumers -- route-casing/route-line/travelled-line -- are
    // all visibility:"none" (drawRouteOverlay below paints the real, visible
    // line straight from pose.route/pose.travelled in screen space instead;
    // see its own comment). Retiling those sources drew zero pixels but cost
    // real time on every frame: setTravelled has no dedup (unlike setRoute)
    // and retiled a strictly-growing whole-trip LineString every frame of a
    // multi-hour film, which was the dominant cost of a long render
    // (per-frame time climbed steadily as the travelled line grew).
    // pose.route/pose.travelled are still
    // sent and still fully determine the frame -- drawRouteOverlay is the
    // only place that reads them now.
    return new Promise(function (resolve, reject) {
      var deadline = Date.now() + budget;
      var settled = false;
      function fail(why) {
        if (settled) return;
        settled = true;
        reject(new Error("scene did not settle within " + budget + "ms: " + why));
      }
      // Success is decided ONLY on "idle", which MapLibre fires at the end of
      // a render that left nothing loading or animating -- so the canvas
      // already shows this pose. Resolving from the timer on
      // areTilesLoaded() alone can screenshot a canvas whose last paint
      // predates the loaded tiles (non-deterministic frames at a 60 ms poll).
      function onIdle() {
        if (settled) return;
        if (!tileFailures.length && map.areTilesLoaded()) {
          settled = true;
          // The ONLY path that resolves with a frame (the fail() path
          // above rejects instead -- capture.py re-renders the whole pose
          // rather than screenshot a half-settled one), so this is the
          // single place the overlay needs to be current before resolve.
          drawRouteOverlay(pose);
          resolve({ labels: projectLabels(pose.marker || null) });
          return;
        }
        map.once("idle", onIdle);
      }
      // "idle" may never fire again once a tile errors, so a timer surfaces
      // tile failures and the deadline; it never declares success.
      function watchdog() {
        if (settled) return;
        if (tileFailures.length) {
          fail(tileFailures.join("; "));
          return;
        }
        if (Date.now() >= deadline) {
          fail("tiles still loading");
          return;
        }
        window.setTimeout(watchdog, SETTLE_POLL_MS);
      }
      // Listen before forcing a repaint, so an unchanged pose (a
      // determinism re-render) still produces the render that fires "idle".
      map.once("idle", onIdle);
      map.triggerRepaint();
      window.setTimeout(watchdog, SETTLE_POLL_MS);
    });
  }

  window.routeStory = {
    init: init,
    renderPose: renderPose,
    ready: ready,
    isReady: function () {
      if (!isReady && map !== null && !loadFired && map.loaded()) {
        // MapLibre fires "load" only from inside a render pass, and schedules
        // a pass only when something is dirty. When the last source finishes
        // between two passes, map.loaded() flips true with no pass to follow,
        // and "load" never comes -- the driver's forced screenshots cannot
        // help, because no frame was requested. A fresh
        // browser can sit the whole ready budget this way, reporting
        // map.loaded() true with no pending frame.
        // A synchronous redraw IS that pass, and it fires "load".
        map.redraw();
      }
      return isReady;
    },
    // What the scene is still waiting for. Called by the capture when
    // readiness times out: without it the driver can only report THAT the
    // scene never settled, which cost two four-hour renders to learn nothing.
    diagnose: function () {
      if (map === null) {
        return { initialised: false };
      }
      return {
        initialised: true,
        loadFired: loadFired,
        labelsLoaded: labelsLoaded,
        styleLoaded: map.isStyleLoaded(),
        tilesLoaded: map.areTilesLoaded(),
        labelCount: labels.length,
        tileFailures: tileFailures.slice(0, 12),
      };
    },
  };
})();
