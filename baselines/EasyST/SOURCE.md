# EasyST source note

- Paper/code reference: <https://github.com/HKUDS/EasyST>, inspected at commit
  `4621c2a472f795df81220e048b4cbbcf10cc1422`.
- The inspected repository does not provide a software license and its student
  source also references a missing local `mlp.py`.  No source file was copied.
- This directory therefore contains a clean BasicTS adaptation of the paper's
  deployable student design: history MLP, learned spatial prompt, calendar
  prompt, residual MLP encoder and a compact feature bottleneck.  It is trained
  directly under the common atmospheric protocol rather than relying on a
  separately trained teacher checkpoint.
