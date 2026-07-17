"""Interactive script to open a robot XML in the MuJoCo viewer."""

import subprocess
import sys
from pathlib import Path


def find_robot_xmls() -> list[Path]:
  """Find all XML files under src/assets/robots."""
  repo_root = Path(__file__).resolve().parent.parent
  robots_dir = repo_root / "src" / "assets" / "robots"
  if not robots_dir.exists():
    print(f"[ERROR] Robot assets directory not found: {robots_dir}")
    sys.exit(1)
  return sorted(robots_dir.rglob("*.xml"))


def prompt_choice(xmls: list[Path]) -> Path:
  """Display a numbered list and ask the user to pick one XML."""
  print("Available robot XML files:")
  for i, xml in enumerate(xmls, start=1):
    print(f"  {i:2d}. {xml}")

  while True:
    choice = input(f"\nSelect a file (1-{len(xmls)}, or q to quit): ").strip()
    if choice.lower() in ("q", "quit", "exit"):
      sys.exit(0)
    if not choice.isdigit():
      print("Please enter a number.")
      continue
    idx = int(choice)
    if idx < 1 or idx > len(xmls):
      print(f"Please enter a number between 1 and {len(xmls)}.")
      continue
    return xmls[idx - 1]


def launch_viewer(xml_path: Path) -> None:
  """Launch the MuJoCo viewer for the selected XML."""
  cmd = [
    sys.executable,
    "-c",
    (
      "import mujoco.viewer; import mujoco; "
      f"m = mujoco.MjModel.from_xml_path('{xml_path}'); "
      "mujoco.viewer.launch(m)"
    ),
  ]
  print(f"\nLaunching viewer for: {xml_path}\n")
  subprocess.run(cmd)


def main() -> None:
  xmls = find_robot_xmls()
  if not xmls:
    print("[ERROR] No XML files found in src/assets/robots.")
    sys.exit(1)

  selected = prompt_choice(xmls)
  launch_viewer(selected)


if __name__ == "__main__":
  main()
