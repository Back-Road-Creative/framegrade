# Security Policy

## Supported versions

Only the latest release receives security fixes. Pin a released `v*` tag (see
the install line in [`README.md`](README.md)); older releases and unreleased
`main` are not supported.

## Reporting a vulnerability

Please report suspected vulnerabilities privately. Do **not** open a public
issue for a security report.

Open a private advisory through GitHub's **Security -> Report a vulnerability**
tab on this repository:
<https://github.com/Back-Road-Creative/framegrade/security/advisories/new>

Please include:

- the affected version,
- a description of the issue and its impact,
- steps to reproduce it,
- any suggested fix.

## What Framegrade touches

- **Local media files.** Framegrade reads and writes video and image files on
  the machine it runs on, and runs `ffmpeg` on them. Treat media from sources
  you do not trust accordingly.
- **Platform credentials.** The `upload` command can use a Google Drive OAuth
  credentials file and token (`--credentials`, `--token`, or `--init-oauth`
  for the one-time browser sign-in), and the YouTube commands read per-brand
  credentials from a credentials directory (`seo.credentials_dir` in the
  config). Keep these files out of version control and out of bug reports.
  The repository's public-tree check rejects tracked files named like
  credentials or tokens.

## Scope

In scope: the code in this repository and its workflows under
`.github/workflows/`.

Out of scope: issues in third-party tools Framegrade calls (such as `ffmpeg`)
or in the platforms it publishes to; report those to their maintainers.
