# CHANGES

Commit-by-commit summary (newest first).

## 🔺 High — dca25cf / 2026-04-27 / add PID controller for movements

### Explanation
- Added PID movement controller with configurable gains/timing/deadzone and episode scoring metrics.
- Added runtime actions for move/tune/benchmark/auto-tune and non-blocking task progression in the main loop.
- Uses higher-rate tracking from `7aafaf5` to improve control response.

## 🔸 Med — 7aafaf5 / 2026-04-27 / make char coord tracking at 60 FPS to improve pid control

### Explanation
- Raised character coordinate tracking from 15 FPS to 60 FPS.
- Raised DXGI capture rate from 30 FPS to 60 FPS to keep sensing and control cadence aligned.
- This made PID movement updates in `dca25cf` more responsive.

## 🔸 Med — aede0f2 / 2026-04-27 / remove continuous save of char coord

### Explanation
- Stopped saving cache on every character coordinate write.
- Stopped saving cache on every track-character toggle update.
- Reduced disk I/O pressure in high-frequency tracking paths.

## 🔸 Med — b352694 / 2026-04-27 / add new fuzzy matched for command palette

### Explanation
- Switched command palette matching to improved fuzzy matcher path.
- Improved action ranking/selection behavior in command input flow.

## 🔸 Med — 65f6eaf / 2026-04-27 / system stats smoothing char coord tracking with smaller window (less computations) lower FPS of char coord calc

### Explanation
- Added smoothing path for sampled system stats and adjusted character tracking window/frequency tradeoffs.
- Reduced per-frame compute pressure by shrinking tracking workload.

## 🔸 Med — f6cea22 / 2026-04-27 / add caching of png encoded img small tweaks to command_palette

### Explanation
- Added cache for PNG-encoded composite image bytes to avoid re-encoding every draw.
- Included command palette usability/performance tweaks in the same change set.

## 🔸 Med — 5c51f43 / 2026-04-26 / minor optimizations

### Explanation
- Minor optimizations.
- Touched `assistant/main.py`.

## 🔸 Med — a4e9222 / 2026-04-26 / move overlay dev dep to another localtion disable dirty optimize

### Explanation
- Moved overlay dev dep to another localtion disable dirty optimize.
- Touched modules: `assistant/`.

## 🔸 Med — 3d167d4 / 2026-04-23 / add process stats to hud add coord tracking

### Explanation
- Added process CPU/memory stats sampling and rendered them in HUD.
- Added initial character coordinate tracking flow and overlay marker updates.

## 🔸 Med — 3b0a750 / 2026-04-22 / add map_graph delete action

### Explanation
- Added action to delete/reset persisted map graph data.
- Connected graph cleanup to runtime command system for quick recovery from bad graph state.

## 🔸 Med — 6149731 / 2026-04-22 / remove motion demo from main

### Explanation
- Removed motion demo from main.
- Touched `assistant/main.py`.

## 🔸 Med — d78a64c / 2026-04-22 / add kalman filtering in motion demo

### Explanation
- Added kalman filtering in motion demo.
- Touched `assistant/main.py`.
- Follow-up to `6149731` on the same `demo` area.

## 🔸 Med — 53d8f32 / 2026-04-22 / Merge branch 'map_graph'

### Explanation
- Merged branch changes into the current line of development.
- Touched `assistant/main.py`.

## 🔸 Med — 1b204d7 / 2026-04-22 / make velocity demo

### Explanation
- Implemented velocity demo.
- Touched `assistant/main.py`.

## 🔸 Med — 976c6d7 / 2026-04-21 / fix various problems with map_graph population

### Explanation
- Fixed various problems with map_graph population.
- Touched modules: `assistant/`, `map_graph/`.

## 🔸 Med — 8e727df / 2026-04-21 / Merge branch 'main' into map_graph

### Explanation
- Merged branch changes into the current line of development.
- Touched modules: `assistant/`.
- Follow-up to `976c6d7` on the same `map_graph` area.

## 🔸 Med — d885630 / 2026-04-21 / make command history toggle

### Explanation
- Implemented command history toggle.
- Touched modules: `assistant/`.

## 🔹 Low — 0adc326 / 2026-04-21 / upd gitignore

### Explanation
- Updated gitignore.
- Updated ignore patterns: `data/`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔹 Low — ba811be / 2026-04-21 / remove unused deps

### Explanation
- Removed unused deps.
- Touched 2 files.

## 🔹 Low — 567178c / 2026-04-21 / upd gitignore

### Explanation
- Updated gitignore.
- Updated ignore patterns: `*.png`, `./data/`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔸 Med — 667fe6d / 2026-04-21 / add ui brects cache

### Explanation
- Added ui brects cache.
- Touched modules: `assistant/`, `mapar/`.

## 🔺 High — 5096438 / 2026-04-21 / add persistent map graph

### Explanation
- Added persistent map graph subsystem (`map_graph` package) with models, builder, metrics, constants, and store layers.
- Stored graph structure plus node images on disk (`data/map_graph`) and wired load/save/drop operations.
- Integrated graph persistence into assistant map capture flow and management actions.

## 🔹 Low — 1f528a2 / 2026-04-21 / update gitignore

### Explanation
- Updated `.gitignore` to exclude `entity_detector.ipynb` and generated `*.png` artifacts.
- Prevents notebook/output files from polluting commits during capture/debug workflows.

## 🔸 Med — 8ecca3e / 2026-04-21 / remove unneeded file

### Explanation
- Removed unneeded file.
- Touched 5 files.

## 🔸 Med — 75c0cbe / 2026-04-21 / implement checked offset calculation

### Explanation
- Implement checked offset calculation.
- Touched modules: `assistant/`.

## 🔸 Med — d5341e4 / 2026-04-21 / action remembers last used args

### Explanation
- Action remembers last used args.
- Touched modules: `assistant/`.

## 🔹 Low — dd0b2d0 / 2026-04-21 / add docs for Rect, add tweaks

### Explanation
- Added docs for Rect, add tweaks.
- Touched `graphics.py`.

## 🔹 Low — 0857864 / 2026-04-21 / Merge branch 'master' into entity_detector

### Explanation
- Merged branch changes into the current line of development.

## 🔸 Med — 19d434a / 2026-04-21 / fix action rendering

### Explanation
- Fixed action rendering.
- Touched modules: `assistant/`.

## 🔸 Med — 8f709d2 / 2026-04-21 / fix actions

### Explanation
- Fixed actions.
- Touched modules: `assistant/`.

## 🔸 Med — 9aa09d9 / 2026-04-21 / add action decorator

### Explanation
- Added action decorator.
- Touched modules: `assistant/`.

## 🔹 Low — 6c4f785 / 2026-04-21 / bump

### Explanation
- Bump.
- Touched `uv.lock`.

## 🔹 Low — c5d676e / 2026-04-21 / gitignore

### Explanation
- Gitignore.
- Updated ignore patterns: ``, ``, `./config.yaml`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔸 Med — 57c871b / 2026-04-21 / add command selection

### Explanation
- Added command selection.
- Touched modules: `assistant/`.

## 🔹 Low — 4df91d7 / 2026-04-21 / change pyside6 to pyside6-essentials dependency (shrink size)

### Explanation
- Change pyside6 to pyside6-essentials dependency (shrink size).
- Touched 2 files.

## 🔹 Low — 9cd92a3 / 2026-04-21 / add config.yaml to gitignore

### Explanation
- Added config.yaml to gitignore.
- Updated ignore patterns: `config.yaml`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔸 Med — 12664fa / 2026-04-21 / add command input

### Explanation
- Added command input.
- Touched modules: `assistant/`.

## 🔸 Med — 23d9fab / 2026-04-20 / upd

### Explanation
- Upd.
- Touched `entity_detector.ipynb`.

## 🔺 High — cb92cac / 2026-04-20 / make entity_detector

### Explanation
- Implemented entity_detector.
- Touched 6 files.

## 🔸 Med — 413e323 / 2026-04-20 / update assistant entry point

### Explanation
- Updated assistant entry point.
- Touched modules: `assistant/`, `mapar/`.

## 🔺 High — c2aded1 / 2026-04-20 / add assistant entry point

### Explanation
- Added assistant entry point.
- Touched modules: `assistant/`.
- Follow-up to `413e323` on the same `assistant` area.

## 🔹 Low — 6cdc099 / 2026-04-20 / remove requirements.txt

### Explanation
- Removed requirements.txt.
- Touched `requirements.txt`.

## 🔸 Med — f735a3e / 2026-04-20 / remove many outdated things

### Explanation
- Removed many outdated things.
- Touched modules: `experiments/`, `hops/`, `imgs/`, `spes/`.
- Follow-up to `6cdc099` on the same `remove` area.

## 🔸 Med — f1c6171 / 2026-04-20 / fix ui and nonui brects detect

### Explanation
- Fixed ui and nonui brects detect.
- Touched modules: `mapar/`, `tests/`.

## 🔹 Low — e1766d5 / 2026-04-20 / add ahk binary to deps

### Explanation
- Added ahk binary to deps.
- Touched 2 files.

## 🔸 Med — 9740aaa / 2026-04-19 / remove mapar dep

### Explanation
- Removed mapar dep.
- Touched modules: `mapar/`, `tests/`.

## 🔸 Med — c2fc562 / 2026-04-18 / fix snail impl, fix type annotations

### Explanation
- Fixed snail impl, fix type annotations.
- Touched modules: `mapar/`, `osdeps/`.

## 🔹 Low — 5005121 / 2026-04-17 / minor fixes

### Explanation
- Minor fixes.
- Touched modules: `cvutils/`.

## 🔹 Low — cb66a3e / 2026-04-17 / remove unused dep

### Explanation
- Removed unused dep.
- Touched 2 files.

## 🔹 Low — ff72ce4 / 2026-04-17 / remove pointless tests

### Explanation
- Removed pointless tests.
- Touched `tests/test_mapar.py`.
- Follow-up to `cb66a3e` on the same `remove` area.

## 🔹 Low — f1a327f / 2026-04-17 / add environment

### Explanation
- Added environment.
- Touched `.env`.

## 🔺 High — 245cdfa / 2026-04-17 / replace d3dshot dep with dxcam remove obsolete files, overlay lib moved out of project sources into ext lib

### Explanation
- Replaced `d3dshot` capture dependency with `dxcam`.
- Moved overlay library out of project tree into external dependency and removed obsolete files.

## 🔹 Low — 3b1ef77 / 2026-04-17 / upd gitignore

### Explanation
- Updated gitignore.
- Updated ignore patterns: `out/`, `spleeter_test/`, `spes_artifacts/`, `pretrained_models/`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔸 Med — 00c0a68 / 2026-04-17 / lol

### Explanation
- Lol.
- Touched modules: `experiments/`, `mapar/`, `src/`, `tests/`.

## 🔸 Med — d4bc94f / 2025-01-05 / add spidertron patrol command

### Explanation
- Added spidertron patrol command.
- Touched modules: `experiments/`, `tests/`.

## 🔹 Low — 47801c2 / 2025-01-04 / update utils

### Explanation
- Updated utils.
- Touched `experiments/2.ipynb`.

## 🔸 Med — 812db9b / 2025-01-04 / update utils

### Explanation
- Updated utils.
- Touched modules: `cvutils/`, `experiments/`.
- Follow-up to `47801c2` on the same `utils` area.

## 🔹 Low — 347ac75 / 2025-01-01 / Merge branch 'master' of https://github.com/aash/factorio-assistant

### Explanation
- Merged branch changes into the current line of development.

## 🔸 Med — 8a38a1d / 2025-01-01 / add implementation for spidertron utilities

### Explanation
- Added implementation for spidertron utilities.
- Touched modules: `experiments/`.

## 🔹 Low — 3171985 / 2024-11-03 / fix deps

### Explanation
- Fixed deps.
- Touched `tests/test_cvutils.py`.

## 🔹 Low — c90df4f / 2024-11-03 / move sclearn dependencies

### Explanation
- Moved sclearn dependencies.
- Touched 2 files.

## 🔹 Low — a279136 / 2024-10-28 / requirements update

### Explanation
- Requirements update.
- Touched `requirements.txt`.

## 🔸 Med — 3b76dbf / 2024-10-28 / improve markings search

### Explanation
- Improve markings search.
- Touched modules: `experiments/`, `tests/`.

## 🔹 Low — 914ae92 / 2024-09-06 / add default config

### Explanation
- Added default config.
- Touched `config.yaml`.

## 🔹 Low — 05e63a0 / 2024-09-06 / merge belt deploy and others deploy

### Explanation
- Merged branch changes into the current line of development.
- Touched `tests/test_mapar.py`.

## 🔹 Low — 631f059 / 2024-09-06 / fixed hotkey handlers, for get_marks add small feature remove tool

### Explanation
- Fixed hotkey handlers, for get_marks add small feature remove tool.
- Touched `common.py`.

## 🔹 Low — d2681bf / 2024-09-06 / fix compare, add str handling

### Explanation
- Fixed compare, add str handling.
- Touched `graphics.py`.

## 🔸 Med — 1963cac / 2024-09-06 / add config serde

### Explanation
- Added config serde.
- Touched `mapar/snail.py`.

## 🔹 Low — eb413da / 2024-08-30 / add exclusion for expreriments

### Explanation
- Added exclusion for expreriments.
- Touched `.gitattributes`.

## 🔹 Low — 354234d / 2024-08-30 / expreriments with cell classification

### Explanation
- Expreriments with cell classification.
- Touched `experiments/2.ipynb`.
- Follow-up to `eb413da` on the same `expreriments` area.

## 🔹 Low — 72e14f6 / 2024-08-30 / test cell classification

### Explanation
- Test cell classification.
- Touched `tests/test_mapar.py`.
- Follow-up to `354234d` on the same `cell` area.

## 🔹 Low — 8e8c04d / 2024-08-30 / add cell classification

### Explanation
- Added cell classification.
- Touched `common.py`.
- Follow-up to `72e14f6` on the same `cell` area.

## 🔹 Low — e6fe92f / 2024-08-30 / add new npext operations

### Explanation
- Added new npext operations.
- Touched `npext.py`.

## 🔹 Low — 6edcee7 / 2024-08-27 / add posterize op

### Explanation
- Added posterize op.
- Touched `npext.py`.

## 🔹 Low — 311036e / 2024-08-26 / fix non-zero mask, non-zero pixels calc

### Explanation
- Fixed non-zero mask, non-zero pixels calc.
- Touched `npext.py`.

## 🔹 Low — b6cf1e8 / 2024-08-25 / cleanup

### Explanation
- Cleanup.
- Touched `experiments/2.ipynb`.

## 🔹 Low — 3ee24ba / 2024-08-25 / add grid color detection

### Explanation
- Added grid color detection.
- Touched `common.py`.

## 🔹 Low — ee8310b / 2024-08-25 / add color conversion utils

### Explanation
- Added color conversion utils.
- Touched `npext.py`.
- Follow-up to `3ee24ba` on the same `color` area.

## 🔹 Low — c99a912 / 2024-08-23 / add new tests

### Explanation
- Added new tests.
- Touched `tests/test_mapar.py`.

## 🔸 Med — d4e56e9 / 2024-08-23 / refactor snail

### Explanation
- Refactor snail.
- Touched `mapar/snail.py`.

## 🔹 Low — d60c122 / 2024-08-23 / remove obsolete overlay related code

### Explanation
- Removed obsolete overlay related code.
- Touched 2 files.

## 🔹 Low — 6650969 / 2024-08-23 / add many experiments

### Explanation
- Added many experiments.
- Touched modules: `experiments/`.

## 🔹 Low — ecc4676 / 2024-08-23 / move graphics related things to untie npext

### Explanation
- Moved graphics related things to untie npext.
- Touched `graphics.py`.

## 🔹 Low — d8ce0f9 / 2024-08-23 / add common utils

### Explanation
- Added common utils.
- Touched `common.py`.

## 🔺 High — 6126117 / 2024-08-23 / add overlay client/server

### Explanation
- Added overlay client/server implementation as a dedicated module.
- Provided foundation for scene-based primitive rendering used by assistant HUD/map overlays.

## 🔹 Low — 78237a9 / 2024-08-23 / add numpy extension

### Explanation
- Added numpy extension.
- Touched `npext.py`.

## 🔹 Low — d1e8630 / 2024-04-25 / wreck overlay

### Explanation
- Wreck overlay.
- Touched `overlay.py`.

## 🔹 Low — 921d788 / 2024-04-24 / add missing dependency

### Explanation
- Added missing dependency.
- Touched `overlay.py`.

## 🔺 High — 01308a7 / 2024-04-24 / new implementation of overlay window for debugging

### Explanation
- New implementation of overlay window for debugging.
- Touched 2 files.

## 🔸 Med — 91d7598 / 2024-04-24 / strip output from py notebooks

### Explanation
- Strip output from py notebooks.
- Touched modules: `experiments/`.

## 🔸 Med — 562a865 / 2024-04-24 / create api for parsing current player coordinates

### Explanation
- Create api for parsing current player coordinates.
- Touched modules: `mapar/`, `tests/`.

## 🔸 Med — a4030d2 / 2024-03-25 / tweaks

### Explanation
- Tweaks.
- Touched modules: `mapar/`, `tests/`.

## 🔹 Low — fe60c97 / 2024-03-24 / add new tests

### Explanation
- Added new tests.
- Touched `tests/test_mapar.py`.

## 🔸 Med — 17758ff / 2024-03-24 / movement and mapping deps

### Explanation
- Movement and mapping deps.
- Touched modules: `mapar/`, `tests/`.

## 🔹 Low — 56acaee / 2024-03-24 / make test configs

### Explanation
- Implemented test configs.
- Touched `conftest.py`.

## 🔹 Low — 704a7b4 / 2024-02-28 / move intermediate files to tmp directory

### Explanation
- Moved intermediate files to tmp directory.
- Touched `tests/test_mapar.py`.

## 🔺 High — e5b311c / 2024-02-27 / add map parser

### Explanation
- Added map parser.
- Touched modules: `mapar/`, `tests/`.

## 🔹 Low — 11eb32b / 2024-02-27 / add test logging configuration

### Explanation
- Added test logging configuration.
- Touched `conftest.py`.

## 🔹 Low — 73223dc / 2024-02-22 / cleanup

### Explanation
- Cleanup.
- Touched `overlay.py`.

## 🔹 Low — d8895bb / 2024-02-22 / move screen utils to separate module

### Explanation
- Moved screen utils to separate module.
- Touched `overlay.py`.

## 🔹 Low — 939539c / 2024-02-22 / move screen utils to separate module

### Explanation
- Moved screen utils to separate module.
- Touched `osdeps/screen.py`.
- Follow-up to `d8895bb` on the same `module` area.

## 🔹 Low — 5ac97f0 / 2024-02-19 / move utilities to separate module

### Explanation
- Moved utilities to separate module.
- Touched `overlay.py`.
- Follow-up to `939539c` on the same `module` area.

## 🔹 Low — c964103 / 2024-02-19 / fix model path

### Explanation
- Fixed model path.
- Touched `hops/hops.py`.

## 🔹 Low — 384447b / 2024-02-19 / update gitignore

### Explanation
- Updated gitignore.
- Updated ignore patterns: `__pycache__`, `logs/**`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔸 Med — 6cc43b5 / 2024-02-19 / add image conversion utils and tests

### Explanation
- Added image conversion utils and tests.
- Touched modules: `cvutils/`, `tests/`.

## 🔹 Low — df1df38 / 2024-02-18 / update gitignore

### Explanation
- Updated gitignore.
- Updated ignore patterns: `D3DShot/**`, `D3DShot/*`, `1.ipynb`, `2.ipynb`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔹 Low — c8f8b60 / 2024-02-18 / update gitignore

### Explanation
- Updated gitignore.
- Updated ignore patterns: `D3DShot/`, `ultralytics/`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔹 Low — 8089298 / 2024-02-18 / add overlay

### Explanation
- Added early overlay module scaffolding.
- Established initial rendering integration path before later client/server refactor (`6126117`).

## 🔹 Low — 24b10c0 / 2024-02-18 / add yolo model to exclusions

### Explanation
- Added yolo model to exclusions.
- Updated ignore patterns: `tmp/**`, `best.pt`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔹 Low — e353dbc / 2024-02-18 / delete main

### Explanation
- Delete main.
- Touched `main.py`.

## 🔹 Low — cff7e34 / 2024-02-18 / add hops service

### Explanation
- Added hops service.
- Touched `hops/hops.py`.

## 🔹 Low — 9227401 / 2024-02-18 / add reference images

### Explanation
- Added reference images.
- Touched modules: `imgs/`.

## 🔹 Low — 77d8ccb / 2024-02-18 / add gitignore

### Explanation
- Added gitignore.
- Updated ignore patterns: ``, `ultralytics/**`, `D3DShot/**`, `out/**`.
- Repository hygiene change; runtime behavior is unaffected.

## 🔹 Low — 40e9ba4 / 2024-02-18 / add spes service

### Explanation
- Added spes service.
- Touched `spes/spes.py`.

## 🔹 Low — e15f3db / 2024-02-18 / add recipes

### Explanation
- Added recipes.
- Touched 2 files.
