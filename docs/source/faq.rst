.. _faq:

==========================
Frequently Asked Questions
==========================


Issues with running diffstruc
=============================

Sometimes, diffstruc installs correctly, but you may encounter issues when running it.
This is often due to missing dependencies or incorrect environment variables.

Stack Size Errors
-----------------

If you encounter an error like

```
  <ERROR> *cmd_run*:stopping due to failed executions
```

This has been encountered by some users when running diffstruc compiled with flang.
The error occurs when the reverse mode gradient calls an assumed size array argument with too large of a size, which causes the flang runtime to fail.

**Solution:** If you are using flang, use the following command to increase the maximum size of assumed size arrays:

```
ulimit -s unlimited
```

This will allow the flang runtime to handle larger arrays and should resolve the issue.
This also works with other compilers that may have similar stack size limitations.
