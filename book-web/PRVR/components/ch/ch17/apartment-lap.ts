/**
 * The lap that w17.1 and w17.2 share.
 *
 * Rusty drives east down the Apartment corridor, turns around, and comes back
 * over ground he has already mapped — the smallest honest loop closure the
 * floorplan allows. The odometry he reports is deliberately *biased*, not just
 * noisy: a 5% long wheel and a 6 mrad list to port, which is what real robots
 * have and what dead reckoning cannot survive.
 */

import { DEFAULT_RBPF_OPTIONS } from '@/lib/filters/rbpf';
import { probToLogOdds, type InverseModelParams, type OccupancyGridOptions } from '@/lib/mapping/occgrid';
import type { OdomDelta } from '@/lib/models/motion';
import { beamAngles, diffDriveStep, type ScanParams } from '@/lib/sim/world';
import type { Pose2 } from '@/lib/geom/se2';

export const N_BEAMS = 20;
export const MAX_RANGE = 6;

export const BEAM_ANGLES = beamAngles({ nBeams: N_BEAMS, fov: 2 * Math.PI });

export const SCAN_PARAMS: ScanParams = {
  nBeams: N_BEAMS,
  fov: 2 * Math.PI,
  maxRange: MAX_RANGE,
  sigma: 0.03,
};

/**
 * Chapter 13's inverse model, widened: β must cover the 18° gap between beams
 * or the raster carves nothing, and α is two cells thick so a wall lands in the
 * cells it actually occupies.
 */
export const INVERSE_MODEL: InverseModelParams = {
  alpha: 0.4,
  beta: (24 * Math.PI) / 180,
  maxRange: MAX_RANGE,
  lOcc: probToLogOdds(0.82),
  lFree: probToLogOdds(0.3),
  l0: 0,
  clamp: 8,
};

/** 30 cm cells: coarse enough that M of them fit in a browser tab. */
export const MAP_OPTS: OccupancyGridOptions = {
  width: 40,
  height: 30,
  cellSize: 0.3,
  origin: { x: 0, y: 0 },
};

export const START_POSE: Pose2 = { x: 1.3, y: 4.4, theta: 0 };

/** Steps in one out-and-back lap. */
export const LAP = 96;

export const RBPF_BASE = DEFAULT_RBPF_OPTIONS;

/** Where Rusty actually is at tick t — noise-free, and never shown to the filter. */
export function driveStep(pose: Pose2, tick: number): Pose2 {
  const t = tick % LAP;
  if (t < 35) return diffDriveStep(pose, 0.28, 0, 1);
  if (t < 47) return diffDriveStep(pose, 0, Math.PI / 12, 1);
  if (t < 82) return diffDriveStep(pose, 0.28, 0, 1);
  return diffDriveStep(pose, 0, Math.PI / 12, 1);
}

/** The wheels' version of that motion: scaled long, and biased to one side. */
export function odometryReading(u: OdomDelta): OdomDelta {
  return {
    rot1: u.rot1 * 1.03 + 0.006,
    trans: u.trans * 1.05,
    rot2: u.rot2 * 1.03 + 0.006,
  };
}
