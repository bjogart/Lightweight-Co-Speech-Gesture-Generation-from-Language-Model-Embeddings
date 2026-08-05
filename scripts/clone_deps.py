import os
import subprocess


def clone_deps():
  for repo in [
    # This fork contains EMAGE code from
    # https://github.com/PantoMatrix/PantoMatrix. That repository is not
    # compatible with Transformers version 5.0.0. This fork adds three
    # `self.post_init()` calls to EMAGE module constructors that see use in this
    # pipeline. Transformers version 5.0.0 is required because Ministral 3 is
    # not compatible with older versions of the transformers library.
    {
      "url": "https://github.com/bjogart/PantoMatrix",
      "commit": "3c3337b4a6960f642738cf1585fe0c07e3f09832",
    },
  ]:
    clone_dep(repo)


def clone_dep(repo):
  if not os.path.exists(repo_dir(repo)):
    subprocess.run(
      ["git", "clone", repo["url"]],
      check=True,
    )
  commit = commit_hash(repo)
  if commit != repo["commit"]:
    print(commit, repo["commit"])
    subprocess.run(["git", "switch", "--detach", repo["commit"]], check=True)
  assert repo["commit"] == commit_hash(repo)


def commit_hash(repo):
  return (
    subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=repo_dir(repo),
      check=True,
      capture_output=True,
    )
    .stdout.decode()
    .strip()
  )


def repo_dir(repo):
  return os.path.basename(repo["url"])
