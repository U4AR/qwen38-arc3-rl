# Results

Train games (5): ar25-0c556536, cd82-fb555c5d, ft09-0d8bbf25, lp85-305b61c3, ls20-9607627b

Held-out games (20): bp35-0a0ad940, cn04-2fe56bfb, dc22-fdcac232, g50t-5849a774, ka59-38d34dbb, lf52-271a04aa, m0r0-492f87ba, r11l-495a7899, re86-8af5384d, s5i5-18d95033, sb26-7fbdac44, sc25-635fd71a, sk48-d8078629, sp80-589a99af, su15-1944f8ab, tn36-ef4dde99, tr87-cd924810, tu93-0768757b, vc33-5430563c, wa30-ee6fef47

## Train games

| policy | levels / episode | episodes with ≥1 level | reward / episode | ARC score | gen tokens / solved level | gen tokens / episode | watch_video calls / episode |
|---|---|---|---|---|---|---|---|
| baseline (10 ep) | 0.80 | 0.70 | 0.95 | 3.22 | 25,423 | 63,746 | 0.80 |
| baseline-terse (10 ep) | 0.80 | 0.70 | 1.00 | 2.92 | 19,989 | 61,370 | 4.30 |

## Held-out games

| policy | levels / episode | episodes with ≥1 level | reward / episode | ARC score | gen tokens / solved level | gen tokens / episode | watch_video calls / episode |
|---|---|---|---|---|---|---|---|
| baseline (14 ep) | 0.14 | 0.14 | 0.16 | 0.47 | 37,571 | 62,744 | 1.57 |
| baseline-terse (14 ep) | 0.00 | 0.00 | 0.00 | 0.00 | – | 61,204 | 3.86 |

## All games

| policy | levels / episode | episodes with ≥1 level | reward / episode | ARC score | gen tokens / solved level | gen tokens / episode | watch_video calls / episode |
|---|---|---|---|---|---|---|---|
| baseline (24 ep) | 0.42 | 0.38 | 0.49 | 1.62 | 27,853 | 63,162 | 1.25 |
| baseline-terse (24 ep) | 0.33 | 0.29 | 0.42 | 1.21 | 19,989 | 61,273 | 4.04 |

## Per game: levels / episode

| game | split | baseline | baseline-terse |
|---|---|---|---|
| ar25-0c556536 | train | 1.00 | 1.00 |
| bp35-0a0ad940 | held-out | 0.00 | 0.00 |
| cd82-fb555c5d | train | 0.50 | 0.00 |
| cn04-2fe56bfb | held-out | 0.50 | 0.00 |
| dc22-fdcac232 | held-out | 0.00 | 0.00 |
| ft09-0d8bbf25 | train | 1.00 | 1.00 |
| g50t-5849a774 | held-out | 0.00 | 0.00 |
| ka59-38d34dbb | held-out | 0.00 | 0.00 |
| lf52-271a04aa | held-out | 0.50 | 0.00 |
| lp85-305b61c3 | train | 1.00 | 1.00 |
| ls20-9607627b | train | 0.50 | 1.00 |
| m0r0-492f87ba | held-out | 0.00 | 0.00 |
