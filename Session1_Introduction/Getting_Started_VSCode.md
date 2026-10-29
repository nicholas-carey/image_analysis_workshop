# Getting started (using VSCode)


## Clone this GitHub repository to your own computer
* Fork this repository: Log in to [GitHub](https://github.com). On the [GitHub image_analysis_workshop](https://github.com/UCSF-CALM/image_analysis_workshop) page find the "Fork" button (near the right top) and click it. Create the fork under your own GitHub name ("your_github_name").  This will create a copy of the GitHub repository under your own name (i.e. you own the copy).
* Start VSCode
* Clone your fork:
    * Click the Source Control icon (icon with 3 circles, in my case on the left side).  Shortcut (on Windows: Ctrl+Shift+G).
    * Click "Clone Repository"
    * Select "Clone from GitHub"
    * Sign in to your GitHub account.
    * Select your Fork of this repository (created above).
    * Select a location where you will store your copy of the repository (suggestions: ~/projects).
    * Open the repository when prompted
    * VSCode will ask "Do you trust the authors of the files in this folder?".  Click "Yes, I trust the authors".  Until you do this, VSCode runs in a restricted mode and features like Git integration will not fully work.
* Add the UCSF_CALM repository as the "upstream" repository:
  * In the Source Control panel, click the "..." (More Actions) menu at the top. Ensure that "Repositories" is shown. Click the "..." next to "image_analysis_workshop".
  * Select "Remote" > "Add Remote...".
  * Enter the URL: 
  ```text
  https://github.com/UCSF-CALM/image_analysis_workshop.git`
  ```
  * Enter "upstream" as the remote name.

> **Mac troubleshooting:** If no Git-related options show up in VSCode (in the Command Palette or in the "..." menus), even though `git` works in your Terminal, VSCode is likely not finding your `git` installation. This happens because apps launched from Finder/Dock do not automatically pick up the same PATH as your Terminal. To fix it:
>   1. In Terminal, run `which git` to find the full path to git (e.g. `/usr/bin/git` or `/opt/homebrew/bin/git`).
>   2. In VSCode, open Settings (Cmd+,), search for "git.path", and enter that full path.
>   3. Restart VSCode.
>
> Alternatively, open the folder from Terminal by typing `code .` inside it; VSCode will then inherit Terminal's environment (including PATH).

## Create a git branch where you will keep your own work
Under the hood, git stores differences (diffs) between versions.  You create a new version by making a "commit".  However, you will want to add your own commits to your own "branch" (so that you can always go back to a pristine state).  To create your own branch, called "my_branch":
* Click on the branch name shown in the bottom-left corner of the VSCode status bar (it will say "main").
* In the dropdown that appears, select "Create new branch...".
* Type "my_branch" and press Enter.

VSCode will switch you to this new branch, which is now shown in the status bar.

Now create a test file and add some text to it:
* In the Explorer panel, create a new file named "test.txt" (File > New File..., or right-click in the Explorer and select "New File...").
* Add some text to it, for example "The first content for my own branch".
* Save the file (Ctrl+S / Cmd+S).

Add this file to the list of files to be committed, and then commit it:
* Click the Source Control icon in the Activity bar (the icon with 3 circles), or press Ctrl+Shift+G (Cmd+Shift+G on Mac).
* Under "Changes" you will see "test.txt" listed.  Hover over it and click the "+" icon to stage it (this is the same as `git add`).
* Type a commit message, for example "The first file added to my_branch is just a test file.", in the message box at the top of the Source Control panel.
* Click the checkmark (✓) button (or press Ctrl+Enter / Cmd+Enter) to commit.

> IMPORTANT:
> Never commit passwords, secret keys, etc. to a git repository.  It is extremely 
> difficult to remove these from the repo completely, and pushing them to a site
> such as GitHub will make them publicly accessible.


You can now push this file to GitHub:
* Click the "..." (More Actions) menu at the top of the Source Control panel and select "Push".
* Since this is the first time you push this branch, VSCode will instead show a "Publish Branch" button in the Source Control panel - click it.  This creates the branch on GitHub and sets up tracking (the same as `git push --set-upstream origin my_branch`).

Occasionally, there will be changes/additions in the "upstream" UCSF-CALM repository.  You can get these into your work branch as follows:
* Click on the branch name in the status bar and switch to "main".
* Click the "..." menu in the Source Control panel, select "Pull, Push" > "Pull from..." and choose the "upstream" remote and the "main" branch (the same as `git pull upstream main`).
* Click on the branch name in the status bar again and switch back to "my_branch".
* Click the "..." menu, select "Branch" > "Merge Branch..." and choose "main" to merge it into your branch.

Make sure not to change files that are in the "main" branch (which could result in merge conflicts).  

## Create a Python environment
Open a Terminal inside VSCode (menu: Terminal > New Terminal, or use the integrated terminal icon).  Execute the following command:
```text
uv sync
```
This will create a (virtual) Python environment as specified in the file "pyproject.toml".  The specified packages will be downloaded and installed.  They are cached, so that if these packages are needed again later, they will not be downloaded again.

Then run the test script using:
```text
uv run src/show_image.py
```

Fun fact, you can actually skip "uv sync", uv will set up the environment also when you use "uv run".  Next time you run the script, it will be much faster.
