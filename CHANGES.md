# CHANGES

Impact legend: 🔹 Low / 🔸 Med / 🔺 High
## 🔺 e4748ef / 2026-04-29 / refactor: extract PidService, refactor pid_controller to take snail, simplify map_scene_geometry

**Explanation**
- Extracted PidService class encapsulating all PID state and snail interaction.
- Added 6 PID events (tick, move_requested, params_updated, benchmark/tune/stop) on SnailEventBus.
- Refactored pid_controller functions to take snail directly instead of ctx: ActionContext.
- Rewired 6 PID action handlers to emit events; removed all PID globals and wrapper functions.
- Main loop now emits SNAIL_PID_TICK each frame to advance background PID tasks.
- Simplified map_scene_geometry to take tile_size: int instead of full map_tiles list.

---


## 🔺 cd01939 / 2026-04-28 / refactor: introduce event bus architecture with leaf/snail separation

**Explanation**
- Added pyventus-based event bus infrastructure (SnailEventBus, LeafEventBus) for decoupling game state from presentation.
- Extracted game state into SnailState and overlay/presentation state into LeafState dataclasses.
- Moved all overlay rendering into leaf/renderers/ with 8 dedicated modules called from the main loop.
- Rewired screenshot actions to emit snail.screenshot.* events handled by Snail service methods.

---

## 🔸 c0e9b0d / 2026-04-28 / move Snail mapar -> assistant

**Explanation**
- Moved the Snail class from the mapar package to the assistant package with backward-compatible re-export.
- Added screenshot capture service methods to Snail (window, non-UI, center).

---

## 🔹 ddde90c / 2026-04-28 / standardize changelog format and add agents workflow guide

**Explanation**
- Established AGENTS.md with workflow rules for maintaining CHANGES.md from git history.
- Standardized impact labeling (🔹 Low / 🔸 Med / 🔺 High) and section formatting.

---

## 🔺 b730695 / 2026-04-28 / refactor map graph services and improve overlay validation

**Explanation**
- Split map-graph responsibilities into new service and indexing modules.
- Changed overlay/map rendering validation flow and scene consistency checks.

---

## 🔺 dca25cf / 2026-04-27 / add PID controller for movements

**Explanation**
- Added PID-based movement controller with tunable gains, timing, deadzone, and stop conditions.
- Added move/tune/benchmark/auto-tune actions with non-blocking task progression in the main loop.

---

## 🔸 7aafaf5 / 2026-04-27 / make char coord tracking at 60 FPS to improve pid control

**Explanation**
- Raised character-coordinate tracking from 15 FPS to 60 FPS.
- Raised DXGI capture from 30 FPS to 60 FPS for faster feedback to movement control.

---

## 🔸 aede0f2 / 2026-04-27 / remove continuous save of char coord

**Explanation**
- Removed immediate cache saves on character-coordinate updates.
- Removed immediate cache saves when toggling coordinate tracking.

---

## 🔸 b352694 / 2026-04-27 / add new fuzzy matched for command palette

**Explanation**
- Added new fuzzy matched for command palette.
- Scope: `assistant/`.

---

## 🔸 65f6eaf / 2026-04-27 / system stats smoothing char coord tracking with smaller window (less computations) lower FPS of char coord calc

**Explanation**
- System stats smoothing char coord tracking with smaller window (less computations) lower FPS of char coord calc.
- Scope: `assistant/`, `mapar/`, `pyproject.toml`.

---

## 🔸 f6cea22 / 2026-04-27 / add caching of png encoded img small tweaks to command_palette

**Explanation**
- Added caching of png encoded img small tweaks to command_palette.
- Scope: `assistant/`, `uv.lock`.

---

## 🔹 5c51f43 / 2026-04-26 / minor optimizations

**Explanation**
- Minor optimizations.
- Touched `assistant/main.py`.

---

## 🔸 a4e9222 / 2026-04-26 / move overlay dev dep to another localtion disable dirty optimize

**Explanation**
- Moved overlay dev dep to another localtion disable dirty optimize.
- Scope: `assistant/`, `pyproject.toml`, `uv.lock`.

---

## 🔸 3d167d4 / 2026-04-23 / add process stats to hud add coord tracking

**Explanation**
- Added process stats to hud add coord tracking.
- Scope: `assistant/`, `map_graph/`, `mapar/`, `pyproject.toml`, `uv.lock`.

---

## 🔸 3b0a750 / 2026-04-22 / add map_graph delete action

**Explanation**
- Added map_graph delete action.
- Touched `assistant/main.py`.

---

## 🔸 6149731 / 2026-04-22 / remove motion demo from main

**Explanation**
- Removed motion demo from main.
- Touched `assistant/main.py`.

---

## 🔸 d78a64c / 2026-04-22 / add kalman filtering in motion demo

**Explanation**
- Added kalman filtering in motion demo.
- Touched `assistant/main.py`.

---

## 🔸 53d8f32 / 2026-04-22 / Merge branch 'map_graph'

**Explanation**
- Merged branch changes.
- Touched `assistant/main.py`.

---

## 🔸 1b204d7 / 2026-04-22 / make velocity demo

**Explanation**
- Implemented velocity demo.
- Touched `assistant/main.py`.

---

## 🔸 976c6d7 / 2026-04-21 / fix various problems with map_graph population

**Explanation**
- Fixed various problems with map_graph population.
- Scope: `assistant/`, `map_graph/`.

---

## 🔸 8e727df / 2026-04-21 / Merge branch 'main' into map_graph

**Explanation**
- Merged branch changes.
- Scope: `assistant/`, `.gitignore`.

---

## 🔸 d885630 / 2026-04-21 / make command history toggle

**Explanation**
- Implemented command history toggle.
- Scope: `assistant/`.

---

## 🔹 0adc326 / 2026-04-21 / upd gitignore

**Explanation**
- Updated gitignore.
- Touched `.gitignore`.

---

## 🔹 ba811be / 2026-04-21 / remove unused deps

**Explanation**
- Removed unused deps.
- Scope: `pyproject.toml`, `uv.lock`.

---

## 🔹 567178c / 2026-04-21 / upd gitignore

**Explanation**
- Updated gitignore.
- Touched `.gitignore`.

---

## 🔸 667fe6d / 2026-04-21 / add ui brects cache

**Explanation**
- Added ui brects cache.
- Scope: `assistant/`, `mapar/`.

---

## 🔺 5096438 / 2026-04-21 / add persistent map graph

**Explanation**
- Added persistent `map_graph` subsystem (models, builder, metrics, constants, store).
- Stored graph JSON + node images on disk and wired load/save/drop lifecycle.

---

## 🔹 1f528a2 / 2026-04-21 / update gitignore

**Explanation**
- Updated `.gitignore` to ignore `entity_detector.ipynb` and generated `*.png` files.

---

## 🔹 8ecca3e / 2026-04-21 / remove unneeded file

**Explanation**
- Removed unneeded file.
- Scope: `entity_detector.ipynb`, `tst.png`.

---

## 🔸 75c0cbe / 2026-04-21 / implement checked offset calculation

**Explanation**
- Implement checked offset calculation.
- Scope: `assistant/`, `entity_detector.py`, `graphics.py`.

---

## 🔸 d5341e4 / 2026-04-21 / action remembers last used args

**Explanation**
- Action remembers last used args.
- Scope: `assistant/`.

---

## 🔹 dd0b2d0 / 2026-04-21 / add docs for Rect, add tweaks

**Explanation**
- Added docs for Rect, add tweaks.
- Touched `graphics.py`.

---

## 🔸 0857864 / 2026-04-21 / Merge branch 'master' into entity_detector

**Explanation**
- Merged branch changes.

---

## 🔸 19d434a / 2026-04-21 / fix action rendering

**Explanation**
- Fixed action rendering.
- Scope: `assistant/`.

---

## 🔸 8f709d2 / 2026-04-21 / fix actions

**Explanation**
- Fixed actions.
- Scope: `assistant/`.

---

## 🔸 9aa09d9 / 2026-04-21 / add action decorator

**Explanation**
- Added action decorator.
- Scope: `assistant/`.

---

## 🔹 6c4f785 / 2026-04-21 / bump

**Explanation**
- Bump.
- Touched `uv.lock`.

---

## 🔹 c5d676e / 2026-04-21 / gitignore

**Explanation**
- Gitignore.
- Touched `.gitignore`.

---

## 🔸 57c871b / 2026-04-21 / add command selection

**Explanation**
- Added command selection.
- Scope: `assistant/`.

---

## 🔹 4df91d7 / 2026-04-21 / change pyside6 to pyside6-essentials dependency (shrink size)

**Explanation**
- Change pyside6 to pyside6-essentials dependency (shrink size).
- Scope: `pyproject.toml`, `uv.lock`.

---

## 🔹 9cd92a3 / 2026-04-21 / add config.yaml to gitignore

**Explanation**
- Added config.yaml to gitignore.
- Touched `.gitignore`.

---

## 🔸 12664fa / 2026-04-21 / add command input

**Explanation**
- Added command input.
- Scope: `assistant/`, `uv.lock`.

---

## 🔹 23d9fab / 2026-04-20 / upd

**Explanation**
- Upd.
- Touched `entity_detector.ipynb`.

---

## 🔺 cb92cac / 2026-04-20 / make entity_detector

**Explanation**
- Implemented entity_detector.
- Scope: `entity_detector.ipynb`, `entity_detector.py`.

---

## 🔸 413e323 / 2026-04-20 / update assistant entry point

**Explanation**
- Updated assistant entry point.
- Scope: `assistant/`, `mapar/`, `pyproject.toml`, `uv.lock`.

---

## 🔺 c2aded1 / 2026-04-20 / add assistant entry point

**Explanation**
- Added assistant entry point.
- Scope: `assistant/`, `pyproject.toml`, `uv.lock`.

---

## 🔹 6cdc099 / 2026-04-20 / remove requirements.txt

**Explanation**
- Removed requirements.txt.
- Touched `requirements.txt`.

---

## 🔹 f735a3e / 2026-04-20 / remove many outdated things

**Explanation**
- Removed many outdated things.
- Scope: `experiments/`, `hops/`, `imgs/`.

---

## 🔸 f1c6171 / 2026-04-20 / fix ui and nonui brects detect

**Explanation**
- Fixed ui and nonui brects detect.
- Scope: `mapar/`, `tests/`, `config.yaml`, `uv.lock`.

---

## 🔹 e1766d5 / 2026-04-20 / add ahk binary to deps

**Explanation**
- Added ahk binary to deps.
- Scope: `pyproject.toml`, `uv.lock`.

---

## 🔺 9740aaa / 2026-04-19 / remove mapar dep

**Explanation**
- Removed mapar dep.
- Scope: `mapar/`, `tests/`, `common.py`, `config.yaml`.

---

## 🔸 c2fc562 / 2026-04-18 / fix snail impl, fix type annotations

**Explanation**
- Fixed snail impl, fix type annotations.
- Scope: `mapar/`, `osdeps/`.

---

## 🔹 5005121 / 2026-04-17 / minor fixes

**Explanation**
- Minor fixes.
- Scope: `cvutils/`, `conftest.py`.

---

## 🔹 cb66a3e / 2026-04-17 / remove unused dep

**Explanation**
- Removed unused dep.
- Scope: `d3dshot.py`, `d3dshot_stub.py`.

---

## 🔹 ff72ce4 / 2026-04-17 / remove pointless tests

**Explanation**
- Removed pointless tests.
- Touched `tests/test_mapar.py`.

---

## 🔹 f1a327f / 2026-04-17 / add environment

**Explanation**
- Added environment.
- Touched `.env`.

---

## 🔺 245cdfa / 2026-04-17 / replace d3dshot dep with dxcam remove obsolete files, overlay lib moved out of project sources into ext lib

**Explanation**
- Replace d3dshot dep with dxcam remove obsolete files, overlay lib moved out of project sources into ext lib.
- Scope: `mapar/`, `overlay_client.py`, `overlay_server.py`.

---

## 🔹 3b1ef77 / 2026-04-17 / upd gitignore

**Explanation**
- Updated gitignore.
- Touched `.gitignore`.

---

## 🔸 00c0a68 / 2026-04-17 / lol

**Explanation**
- Lol.
- Scope: `experiments/`, `mapar/`, `src/`, `.python-version`, `README.md`.

---

## 🔹 d4bc94f / 2025-01-05 / add spidertron patrol command

**Explanation**
- Added spidertron patrol command.
- Scope: `experiments/`, `tests/`, `common.py`.

---

## 🔹 47801c2 / 2025-01-04 / update utils

**Explanation**
- Updated utils.
- Touched `experiments/2.ipynb`.

---

## 🔹 812db9b / 2025-01-04 / update utils

**Explanation**
- Updated utils.
- Scope: `cvutils/`, `experiments/`, `overlay_server.py`, `requirements.txt`.

---

## 🔸 347ac75 / 2025-01-01 / Merge branch 'master' of https://github.com/aash/factorio-assistant

**Explanation**
- Merged branch changes.

---

## 🔹 8a38a1d / 2025-01-01 / add implementation for spidertron utilities

**Explanation**
- Added implementation for spidertron utilities.
- Scope: `experiments/`, `.gitignore`.

---

## 🔹 3171985 / 2024-11-03 / fix deps

**Explanation**
- Fixed deps.
- Touched `tests/test_cvutils.py`.

---

## 🔹 c90df4f / 2024-11-03 / move sclearn dependencies

**Explanation**
- Moved sclearn dependencies.
- Scope: `common.py`, `sclearn_deps.py`.

---

## 🔹 a279136 / 2024-10-28 / requirements update

**Explanation**
- Requirements update.
- Touched `requirements.txt`.

---

## 🔹 3b76dbf / 2024-10-28 / improve markings search

**Explanation**
- Improve markings search.
- Scope: `experiments/`, `tests/`, `common.py`, `overlay_server.py`.

---

## 🔹 914ae92 / 2024-09-06 / add default config

**Explanation**
- Added default config.
- Touched `config.yaml`.

---

## 🔸 05e63a0 / 2024-09-06 / merge belt deploy and others deploy

**Explanation**
- Merged branch changes.
- Touched `tests/test_mapar.py`.

---

## 🔹 631f059 / 2024-09-06 / fixed hotkey handlers, for get_marks add small feature remove tool

**Explanation**
- Fixed hotkey handlers, for get_marks add small feature remove tool.
- Touched `common.py`.

---

## 🔹 d2681bf / 2024-09-06 / fix compare, add str handling

**Explanation**
- Fixed compare, add str handling.
- Touched `graphics.py`.

---

## 🔸 1963cac / 2024-09-06 / add config serde

**Explanation**
- Added config serde.
- Touched `mapar/snail.py`.

---

## 🔹 eb413da / 2024-08-30 / add exclusion for expreriments

**Explanation**
- Added exclusion for expreriments.
- Touched `.gitattributes`.

---

## 🔹 354234d / 2024-08-30 / expreriments with cell classification

**Explanation**
- Expreriments with cell classification.
- Touched `experiments/2.ipynb`.

---

## 🔹 72e14f6 / 2024-08-30 / test cell classification

**Explanation**
- Test cell classification.
- Touched `tests/test_mapar.py`.

---

## 🔹 8e8c04d / 2024-08-30 / add cell classification

**Explanation**
- Added cell classification.
- Touched `common.py`.

---

## 🔹 e6fe92f / 2024-08-30 / add new npext operations

**Explanation**
- Added new npext operations.
- Touched `npext.py`.

---

## 🔹 6edcee7 / 2024-08-27 / add posterize op

**Explanation**
- Added posterize op.
- Touched `npext.py`.

---

## 🔹 311036e / 2024-08-26 / fix non-zero mask, non-zero pixels calc

**Explanation**
- Fixed non-zero mask, non-zero pixels calc.
- Touched `npext.py`.

---

## 🔹 b6cf1e8 / 2024-08-25 / cleanup

**Explanation**
- Cleanup.
- Touched `experiments/2.ipynb`.

---

## 🔹 3ee24ba / 2024-08-25 / add grid color detection

**Explanation**
- Added grid color detection.
- Touched `common.py`.

---

## 🔹 ee8310b / 2024-08-25 / add color conversion utils

**Explanation**
- Added color conversion utils.
- Touched `npext.py`.

---

## 🔹 c99a912 / 2024-08-23 / add new tests

**Explanation**
- Added new tests.
- Touched `tests/test_mapar.py`.

---

## 🔸 d4e56e9 / 2024-08-23 / refactor snail

**Explanation**
- Refactor snail.
- Touched `mapar/snail.py`.

---

## 🔹 d60c122 / 2024-08-23 / remove obsolete overlay related code

**Explanation**
- Removed obsolete overlay related code.
- Scope: `overlay.py`, `overlay_old.py`.

---

## 🔹 6650969 / 2024-08-23 / add many experiments

**Explanation**
- Added many experiments.
- Scope: `experiments/`.

---

## 🔹 ecc4676 / 2024-08-23 / move graphics related things to untie npext

**Explanation**
- Moved graphics related things to untie npext.
- Touched `graphics.py`.

---

## 🔹 d8ce0f9 / 2024-08-23 / add common utils

**Explanation**
- Added common utils.
- Touched `common.py`.

---

## 🔺 6126117 / 2024-08-23 / add overlay client/server

**Explanation**
- Added overlay client/server implementation for scene-based primitive rendering.

---

## 🔹 78237a9 / 2024-08-23 / add numpy extension

**Explanation**
- Added numpy extension.
- Touched `npext.py`.

---

## 🔹 d1e8630 / 2024-04-25 / wreck overlay

**Explanation**
- Wreck overlay.
- Touched `overlay.py`.

---

## 🔹 921d788 / 2024-04-24 / add missing dependency

**Explanation**
- Added missing dependency.
- Touched `overlay.py`.

---

## 🔺 01308a7 / 2024-04-24 / new implementation of overlay window for debugging

**Explanation**
- New implementation of overlay window for debugging.
- Scope: `overlay.py`, `overlay_old.py`.

---

## 🔹 91d7598 / 2024-04-24 / strip output from py notebooks

**Explanation**
- Strip output from py notebooks.
- Scope: `experiments/`, `.gitignore`.

---

## 🔸 562a865 / 2024-04-24 / create api for parsing current player coordinates

**Explanation**
- Create api for parsing current player coordinates.
- Scope: `mapar/`, `tests/`, `common.py`, `d3dshot.py`.

---

## 🔸 a4030d2 / 2024-03-25 / tweaks

**Explanation**
- Tweaks.
- Scope: `mapar/`, `tests/`, `conftest.py`, `requirements.txt`.

---

## 🔹 fe60c97 / 2024-03-24 / add new tests

**Explanation**
- Added new tests.
- Touched `tests/test_mapar.py`.

---

## 🔸 17758ff / 2024-03-24 / movement and mapping deps

**Explanation**
- Movement and mapping deps.
- Scope: `mapar/`, `tests/`, `common.py`.

---

## 🔹 56acaee / 2024-03-24 / make test configs

**Explanation**
- Implemented test configs.
- Touched `conftest.py`.

---

## 🔹 704a7b4 / 2024-02-28 / move intermediate files to tmp directory

**Explanation**
- Moved intermediate files to tmp directory.
- Touched `tests/test_mapar.py`.

---

## 🔺 e5b311c / 2024-02-27 / add map parser

**Explanation**
- Added map parser.
- Scope: `mapar/`, `tests/`.

---

## 🔹 11eb32b / 2024-02-27 / add test logging configuration

**Explanation**
- Added test logging configuration.
- Touched `conftest.py`.

---

## 🔹 73223dc / 2024-02-22 / cleanup

**Explanation**
- Cleanup.
- Touched `overlay.py`.

---

## 🔹 d8895bb / 2024-02-22 / move screen utils to separate module

**Explanation**
- Moved screen utils to separate module.
- Touched `overlay.py`.

---

## 🔹 939539c / 2024-02-22 / move screen utils to separate module

**Explanation**
- Moved screen utils to separate module.
- Touched `osdeps/screen.py`.

---

## 🔹 5ac97f0 / 2024-02-19 / move utilities to separate module

**Explanation**
- Moved utilities to separate module.
- Touched `overlay.py`.

---

## 🔹 c964103 / 2024-02-19 / fix model path

**Explanation**
- Fixed model path.
- Touched `hops/hops.py`.

---

## 🔹 384447b / 2024-02-19 / update gitignore

**Explanation**
- Updated gitignore.
- Touched `.gitignore`.

---

## 🔹 6cc43b5 / 2024-02-19 / add image conversion utils and tests

**Explanation**
- Added image conversion utils and tests.
- Scope: `cvutils/`, `tests/`.

---

## 🔹 df1df38 / 2024-02-18 / update gitignore

**Explanation**
- Updated gitignore.
- Touched `.gitignore`.

---

## 🔹 c8f8b60 / 2024-02-18 / update gitignore

**Explanation**
- Updated gitignore.
- Touched `.gitignore`.

---

## 🔹 8089298 / 2024-02-18 / add overlay

**Explanation**
- Added overlay.
- Touched `overlay.py`.

---

## 🔹 24b10c0 / 2024-02-18 / add yolo model to exclusions

**Explanation**
- Added yolo model to exclusions.
- Touched `.gitignore`.

---

## 🔹 e353dbc / 2024-02-18 / delete main

**Explanation**
- Delete main.
- Touched `main.py`.

---

## 🔹 cff7e34 / 2024-02-18 / add hops service

**Explanation**
- Added hops service.
- Touched `hops/hops.py`.

---

## 🔹 9227401 / 2024-02-18 / add reference images

**Explanation**
- Added reference images.
- Scope: `imgs/`.

---

## 🔹 77d8ccb / 2024-02-18 / add gitignore

**Explanation**
- Added gitignore.
- Touched `.gitignore`.

---

## 🔹 40e9ba4 / 2024-02-18 / add spes service

**Explanation**
- Added spes service.
- Touched `spes/spes.py`.

---

## 🔹 e15f3db / 2024-02-18 / add recipes

**Explanation**
- Added recipes.
- Scope: `main.py`, `recipes.yml`.

