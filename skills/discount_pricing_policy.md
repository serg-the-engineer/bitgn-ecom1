# Skill: Discount And Pricing Policy

Use this for discounts, service recovery, price changes, manager approvals, and policy caps.

Workflow:
1. Identify the target basket or pricing object.
2. Discover the relevant policy and authority evidence in the runtime.
3. Compute the allowed amount or percentage from policy, not from memory.
4. If authorization is missing or unverifiable, prefer a security denial or clarification
   according to the policy risk.
5. Mutate only after target, policy, cap, and issuer are all verified.
6. Verify the resulting state before reporting success.

Do not encode specific people, stores, basket ids, discount values, or policy file paths.
