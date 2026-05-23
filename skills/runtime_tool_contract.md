# Skill: Runtime Tool Contract

Use this for every tool call.

Rules:
1. Runtime tools are not local repository files.
2. Use the explicit runtime tool surface and absolute runtime paths.
3. For database queries, use the runtime SQL executable path.
4. To inspect a workspace document, use read/search/list/tree instead of shelling out to
   a local file utility.
5. If a path is rejected, adapt to the runtime contract instead of retrying local paths.

Do not copy command paths, argument sequences, or query literals from prior tasks into
this skill.
