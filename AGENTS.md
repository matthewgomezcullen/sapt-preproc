Read background literature from the `md` files, not the original `pdf`s.

Follow test-driven development. After planning, write tests first. Wait for me to review the tests before developing.

We want to reduce the amount of bespoke code as much as possible, relying on established open-source libraries instead.

The experiment will run on HPC clusters that only I have access to. Save large end-to-end tests for the cluster. My local computer (that you have access to) runs on macOS.

In chat, write equations following TeX syntax (\mu instead of μ) and use escaped parentheses and fenced math blocks to delimit your equations, so they render correctly.

`test_protonate` takes a long time to run. Do not run these without my permission. Instead, run `pytest tests --fast`

For changes that only touch encoding, use `pytest tests --encode`
