# Skill: Inventory Availability

Use this for stock, availability, store, branch, and city-wide aggregation tasks.

Workflow:
1. Resolve the exact catalog variant before checking availability.
2. Identify the complete relevant store or branch scope.
3. Include zero-stock branches when the task asks for complete coverage.
4. Aggregate only after each candidate record is verified.
5. Before completion, verify the final total against the inspected records.

Use runtime tools to discover current records. Do not encode any fixed city, store,
SKU, date, or product-specific path in this skill.
