# Measured Slayers 2 UI - from real captures, not guides

Read off full-frame WGC captures at 2560x1600. Every online source for this
game was checked and discarded: one site published an "Official PC bindings"
table that does not exist on the actual Roblox game page. So these are
observations, with the frame they came from, not claims from a wiki.

Frame: 2560x1600. `GetClientRect` reports 2560x1571 - **DPI scaling is
active**, so window-relative coordinates need a 1.0185 vertical correction.
The WGC frame is the source of truth for pixel positions.

## What I got wrong, and what is actually true

| I assumed | Reality |
|---|---|
| Skill keys are `1 2 3` | **`Z X C V B`** — five abilities on that row, plus `F` on a separate slot. Config had `["1","2","3"]`, which is wrong. |
| M1 is Space or a key | **Left mouse button**, and there is a `Dash = Q + WASD` binding shown bottom-right. |
| Health bar is a single wide red bar | **Segmented.** The boss bar is ~4 separate red chunks with dark gaps. The player bar is one solid bar. |
| Boss bar is bottom-left style | **Boss bar is top-centre**, under the boss name. Player bar is **bottom-left**. |
| Windows Graphics Capture is exotic | **It is the only thing that can see the game while Chrome is in front.** 70-77 fps, and it bypasses vsync. |

## Measured positions (fractions of frame, 2560x1600)

### Player health - bottom-left
- Text `940 / 1432` around x 0.075-0.105, y ~0.755
- The bar itself: solid red, x ~0.013-0.105, y ~0.775, height ~10px
- **This is a plain left-anchored fill.** Fraction = red width / full width.

### Boss health - top-centre, under the boss name
- Boss name + title (`The Sound Hashira` / `Tengai`) at y ~0.115
- Numeric HP `640.6 / 3000` at x ~0.265-0.305, y ~0.152
- The bar: x ~0.263-0.318, y ~0.165, height ~8px
- **SEGMENTED into ~4 chunks** with dark gaps. A naive "find widest red
  component" picks one chunk and reports a bogus fraction.
- Drop table directly beneath: `Exp +1,000`, `Wen +450`, then item rows.

### Target enemy health - floating, above the enemy
- Name (`Tengai Uzui`) and a red bar directly under it
- Position varies with the enemy, so this cannot use a fixed region.
- Must be found by scanning the middle band of the screen.

### Skill bar - bottom-centre
- Two rows. Upper row: five square ability icons labelled `Z X C V B`,
  with a separate `F` slot to the left. y ~0.68-0.70
- Lower row: five circular item slots labelled `1 2 3 4 5`. y ~0.74-0.78
- `x1.69 Block Regen` status text sits just above the ability row
- A purple/magenta stamina bar sits above that, x ~0.38-0.42, y ~0.62

### Other confirmed UI
- Top-left: chat, `Say hi to everyone playing now!`
- Top-centre: compass ribbon with `N / 45 / 135 / 225 / W`
- Top-right: minimap, `Iceveil Settlement`, `Shrine`, `Serpent Trainee`
- Bottom-right: `Dash  Q + WASD`, Zen `57,060`
- Bottom-left: level `13886 / 21099`, `Lv 192`, stat grid
- `itzhimanshugt 2,359.39 (78%)` = the player's own bar, floating near centre
- `Missing Sound` - a red Roblox error, not a game element
- Mastery bars: `Serpent Mastery Lv 138`, `Sword Mastery Lv 206`

## Consequence for detection

Three detector changes are required, and all three are because the bars are
not what the generic code assumed:

1. **Segmented bars need a span, not a component.** Measure leftmost-to-
   rightmost red pixel across the bar's row, then divide by the full track
   width. The existing `BarDetector.find_track` already does the span thing
   for the fishing bar; `HealthReader` does not, and it will read a
   quarter-full four-segment boss bar as "empty".
2. **Fixed regions will not work for the target bar.** It moves. It has to
   be found by scanning, or the boss macro simply cannot see it.
3. **Ability keys are Z X C V B, not 1 2 3.** Config is wrong and pressing
   the right numbers does nothing at all.

## How these were measured

```powershell
python tools/probe_window.py                    # full frame + fps
python tools/zoom.py X,Y,W,H --scale 4          # crop and upscale
python tools/measure_ui.py --json out.json      # all elements, one grab
```

`measure_ui.py` searches generous fractional zones and keeps the widest
match. It found nothing in one run because the game state had changed - the
boss UI only exists during a boss fight. Run it *during* the thing you want
measured.
