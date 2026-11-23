# Domain Documentation (Multi-Context)

This repo uses **multi-context** domain documentation.

**Layout:**
- Root `GLOSSARY-MAP.md` maps feature areas to their context files
- Per-area `GLOSSARY.md` files in `docs/area/<area>/` define domain terms for each feature area
- Architecture Decision Records (ADRs) live in `docs/adr/`

**Consumer Rules:**
- Read `GLOSSARY-MAP.md` first to understand the area context
- Then read the relevant `GLOSSARY.md` for that area's terminology
- ADRs in `docs/adr/` capture architectural decisions and their rationale
- New terms should be added to the appropriate area's `GLOSSARY.md`
- ADRs should follow the template in `docs/adr/templates/`

**Purpose:**
- Multi-context allows different feature areas to have independent terminology
- The GLOSSARY-MAP provides a single entry point to navigate all area-specific glossaries
- ADRs provide a historical record of architectural decisions