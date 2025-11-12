import re
import subprocess

from InquirerPy import inquirer
from rich.console import Console

console = Console()

# show recent commits (with tags)
console.print("[bold cyan]Recent commits:[/bold cyan]")
commits = subprocess.check_output(
    [
        "git",
        "log",
        '--pretty=format:"%h %s %d"',
        "-n",
        "20",
    ],
    text=True,
).splitlines()

commit_choice = inquirer.select(  # type: ignore
    message="Select commit to tag:", choices=commits
).execute()
commit_sha = commit_choice.split()[0][1:]


def suggest_next_tag():
    tags_output = subprocess.check_output(["git", "tag", "--list"], text=True)
    tags = tags_output.strip().splitlines()

    if not tags:
        return "v0.1.0"

    # Extract semantic versions like v1.2.3
    versions = [re.search(r"v(\d+)\.(\d+)\.(\d+)", t) for t in tags]
    versions = [tuple(map(int, v.groups())) for v in versions if v]

    if not versions:
        return "v0.1.0"

    versions.sort()
    major, minor, patch = versions[-1]  # highest version
    patch += 1

    return f"v{major}.{minor}.{patch}"


suggested_tag = suggest_next_tag()
new_tag = inquirer.text(message="Enter new tag:", default=suggested_tag).execute()  # type: ignore

# create tag
subprocess.run(["git", "tag", "-a", new_tag, commit_sha, "-m", f"Release {new_tag}"])
console.print(
    f"[bold yellow]Created tag {new_tag} on commit {commit_sha}[/bold yellow]"
)

# push tag
push_confirm = inquirer.confirm(message="Push tag to remote?", default=True).execute()  # type: ignore
if push_confirm:
    subprocess.run(["git", "push", "origin", new_tag])
    console.print(f"[bold green]Tag {new_tag} pushed successfully![/bold green]")
else:
    console.print(
        "[bold red]Tag not pushed. Remember to push it manually to trigger the GitHub Action.[/bold red]"
    )
