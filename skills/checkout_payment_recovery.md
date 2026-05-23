# Skill: Checkout And Payment Recovery

Use this for checkout, basket, payment authentication, refund, return, and recovery tasks.

Workflow:
1. Identify the target commerce object and current state.
2. Read relevant workflow or policy material before side-effecting operations.
3. For ambiguous ownership, active basket, or payment state, clarify instead of guessing.
4. For sensitive payment or checkout recovery, run the authorization decision before mutation.
5. After any mutation, verify state before reporting success.

Never use a prior task's basket, customer, payment, or order id as a reusable rule.
