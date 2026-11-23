# Issue Tracker (Local Markdown)

This repo uses local markdown files for issue tracking. Issues are tracked as markdown files under `.scratch/<feature>/` in this repository.

**Workflow:**
- New issues are created as markdown files under `.scratch/<feature>/`
- Each issue file should contain a YAML frontmatter block with title, status, and labels
- Triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`
- Issues can be referenced and managed using the triage skill

**Commands:**
- Create new issue: `.scratch/<feature>/issue-title.md`
- List issues: `ls .scratch/`
- Triage an issue: update its status and labels in the markdown file

This setup is ideal for solo projects or repositories without a remote issue tracker.