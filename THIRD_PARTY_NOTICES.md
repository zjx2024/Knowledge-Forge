# Third-party notices

Knowledge Forge's browser graph explorer uses `@cosmos.gl/graph` 3.4.0,
distributed under the MIT License. Its JavaScript bundle also contains the
package's transitive runtime dependencies. The exact dependency versions are
locked in `webui/package-lock.json`.

The corresponding license texts are retained in
`knowledge_forge/static/graph-explorer.LICENSE.txt`, generated from the locked
dependency tree as part of the frontend build review. Knowledge Forge does not
embed or depend on the separately licensed Cosmograph application or
`@cosmograph/*` product libraries.
