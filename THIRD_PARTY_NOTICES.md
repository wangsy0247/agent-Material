# Third-party notices

Agent Material is licensed under Apache-2.0. Components listed below retain
their own copyrights and licenses; the project license does not replace them.

## Vendored browser assets

| Component | Version | License | Location |
| --- | --- | --- | --- |
| DOMPurify | 3.2.4 | Apache-2.0 OR MPL-2.0 | `frontend/static/vendor/dompurify/` |
| Marked | 12.0.2 | MIT | `frontend/static/vendor/marked/` |
| Highlight.js | 11.9.0 | BSD-3-Clause | `frontend/static/vendor/highlight.js/` |

The version and license above are taken from the license headers embedded in
the vendored files. When updating an asset, update this table and retain its
license header.

## MinerU

PDF parsing can use `mineru-open-sdk`. MinerU is distributed under the
**MinerU Open Source License**, which is based on Apache-2.0 and contains
additional conditions. Deployers must review the license for the exact SDK
version they install, including any service attribution or commercial-use
requirements:

<https://github.com/opendatalab/MinerU/blob/master/LICENSE.md>

## Materials Project

The optional MCP integration queries Materials Project using a key supplied by
the user. Materials Project data is not bundled with this repository. Users
are responsible for protecting their API key and providing the attribution and
database-version citation required by Materials Project:

<https://docs.materialsproject.org/downloading-data/how-do-i-download-the-materials-project-database>

## Python dependencies

Python packages are installed from their upstream distributions and retain
their respective licenses. Dependency versions used by CI and releases must be
recorded in the repository lock file. Before a release, run the license audit
documented in `CONTRIBUTING.md` and investigate unknown or incompatible entries.

## Provenance rule

Code copied or substantially adapted from another project must preserve its
copyright notice and license. Architecture inspiration alone should be named
in the README acknowledgements, without implying upstream endorsement.
