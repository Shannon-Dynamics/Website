/**
 * The book's canonical structure: the single source of truth for chapter
 * numbers, slugs, and part membership. The landing page, the sidebar ordering
 * in content/chapters/meta.json, and every cross-chapter link agree with this.
 */

export interface ChapterEntry {
  n: number;
  slug: string;
  title: string;
  blurb: string;
  difficulty: 'Foundational' | 'Intermediate' | 'Advanced';
}

export interface PartEntry {
  id: string;
  title: string;
  chapters: ChapterEntry[];
}

export const PARTS: PartEntry[] = [
  {
    id: 'PART I',
    title: 'Foundations — The Robot and Its Uncertainty',
    chapters: [
      {
        n: 1,
        slug: 'ch01-robot-that-doubts',
        title: 'The Robot That Doubts',
        blurb: 'Why a single best guess is not enough, and what a belief buys you instead.',
        difficulty: 'Foundational',
      },
      {
        n: 2,
        slug: 'ch02-probability',
        title: 'Probability: The Language of Uncertainty',
        blurb: 'Bayes rule, Gaussians in one and many dimensions, moments and canonical form.',
        difficulty: 'Foundational',
      },
      {
        n: 3,
        slug: 'ch03-geometry-of-motion',
        title: 'The Geometry of Motion',
        blurb: 'Frames, rotations, SE(2) and SE(3), exponential coordinates, and uncertainty on manifolds.',
        difficulty: 'Foundational',
      },
      {
        n: 4,
        slug: 'ch04-rusty-and-sensors',
        title: 'Rusty, Sensors, and the Simulator',
        blurb: 'Wheel odometry, LiDAR physics, and the two worlds every later chapter runs in.',
        difficulty: 'Foundational',
      },
    ],
  },
  {
    id: 'PART II',
    title: 'The Bayes Filter Family',
    chapters: [
      {
        n: 5,
        slug: 'ch05-bayes-filter',
        title: 'The Bayes Filter',
        blurb: 'The recursion behind every estimator in the book: sensing sharpens, moving smears.',
        difficulty: 'Foundational',
      },
      {
        n: 6,
        slug: 'ch06-kalman-filters',
        title: 'Kalman Filters',
        blurb: 'The linear-Gaussian world, the gain as precision-weighted trust, and the information form.',
        difficulty: 'Intermediate',
      },
      {
        n: 7,
        slug: 'ch07-ekf-ukf-manifolds',
        title: 'Beyond Linearity: EKF, UKF, and Manifolds',
        blurb: 'Where linearization lies, what sigma points fix, and how modern filters live on Lie groups.',
        difficulty: 'Advanced',
      },
      {
        n: 8,
        slug: 'ch08-nonparametric-filters',
        title: 'Nonparametric Filters',
        blurb: 'Histograms, importance sampling, particle filters, and the art of resampling.',
        difficulty: 'Intermediate',
      },
    ],
  },
  {
    id: 'PART III',
    title: 'Probabilistic Models',
    chapters: [
      {
        n: 9,
        slug: 'ch09-motion-models',
        title: 'Probabilistic Motion Models',
        blurb: 'Velocity and odometry models, the banana distribution, and noise on the manifold.',
        difficulty: 'Intermediate',
      },
      {
        n: 10,
        slug: 'ch10-sensor-models',
        title: 'Probabilistic Sensor Models',
        blurb: 'Beam mixtures, likelihood fields, landmarks, and the cost of pretending beams are independent.',
        difficulty: 'Intermediate',
      },
    ],
  },
  {
    id: 'PART IV',
    title: 'Localization',
    chapters: [
      {
        n: 11,
        slug: 'ch11-localization-gaussian',
        title: 'Localization I: Tracking with Gaussians',
        blurb: 'EKF localization, data association, and how one wrong match poisons a filter.',
        difficulty: 'Intermediate',
      },
      {
        n: 12,
        slug: 'ch12-localization-global',
        title: 'Localization II: Global Localization',
        blurb: 'Grid localization, MCL, and recovering from being kidnapped.',
        difficulty: 'Intermediate',
      },
    ],
  },
  {
    id: 'PART V',
    title: 'Mapping and SLAM',
    chapters: [
      {
        n: 13,
        slug: 'ch13-occupancy-grids',
        title: 'Occupancy Grid Mapping',
        blurb: 'Log-odds mapping with known poses, and where per-cell independence betrays you.',
        difficulty: 'Intermediate',
      },
      {
        n: 14,
        slug: 'ch14-ekf-slam',
        title: 'The SLAM Problem and EKF SLAM',
        blurb: 'Why correlations are the map, and the two flaws that ended the filtering era.',
        difficulty: 'Advanced',
      },
      {
        n: 15,
        slug: 'ch15-factor-graphs',
        title: 'SLAM as Least Squares: Factor Graphs',
        blurb: 'The modern backbone: MAP inference as sparse nonlinear least squares.',
        difficulty: 'Advanced',
      },
      {
        n: 16,
        slug: 'ch16-scan-matching',
        title: 'Scan Matching and Pose-Graph SLAM',
        blurb: 'ICP, NDT, loop closure, and a complete 2-D SLAM system built by hand.',
        difficulty: 'Advanced',
      },
      {
        n: 17,
        slug: 'ch17-fastslam',
        title: 'FastSLAM and Rao-Blackwellization',
        blurb: 'Sample the trajectory, solve the map in closed form — one particle, one universe.',
        difficulty: 'Advanced',
      },
      {
        n: 18,
        slug: 'ch18-visual-slam',
        title: 'Visual and Visual-Inertial SLAM',
        blurb: 'Cameras as probabilistic sensors, IMU preintegration, and marginalization.',
        difficulty: 'Advanced',
      },
      {
        n: 19,
        slug: 'ch19-map-representations',
        title: 'Modern Map Representations',
        blurb: 'Octrees, TSDFs, and distance fields — mapping as recursive estimation in disguise.',
        difficulty: 'Intermediate',
      },
    ],
  },
  {
    id: 'PART VI',
    title: 'Planning and Acting under Uncertainty',
    chapters: [
      {
        n: 20,
        slug: 'ch20-motion-planning',
        title: 'Motion Planning: From Geometry to Probability',
        blurb: 'Configuration space, A*, and sampling-based planners that are only probably complete.',
        difficulty: 'Intermediate',
      },
      {
        n: 21,
        slug: 'ch21-mdps',
        title: 'Decision Making I: MDPs',
        blurb: 'From plans to policies, and the Bellman equation that produces them.',
        difficulty: 'Intermediate',
      },
      {
        n: 22,
        slug: 'ch22-pomdps',
        title: 'Decision Making II: POMDPs',
        blurb: 'Planning in belief space, alpha-vectors, and why robots hug walls.',
        difficulty: 'Advanced',
      },
      {
        n: 23,
        slug: 'ch23-mppi',
        title: 'Stochastic MPC: MPPI and Friends',
        blurb: 'Control by sampling: a thousand imagined futures, reweighted every frame.',
        difficulty: 'Advanced',
      },
      {
        n: 24,
        slug: 'ch24-exploration',
        title: 'Exploration and Active SLAM',
        blurb: 'Where should the robot go in order to learn? Information gain, made concrete.',
        difficulty: 'Advanced',
      },
    ],
  },
  {
    id: 'PART VII',
    title: 'Frontiers and Integration',
    chapters: [
      {
        n: 25,
        slug: 'ch25-learning',
        title: 'Learning in the Loop',
        blurb: 'Learned sensor models, calibration, and differentiable filters — inside the Bayesian frame.',
        difficulty: 'Advanced',
      },
      {
        n: 26,
        slug: 'ch26-capstone',
        title: 'Capstone: A Complete Autonomous Robot',
        blurb: 'Everything wired together: explore, map, plan, control, recover, repeat.',
        difficulty: 'Advanced',
      },
    ],
  },
];

export const ALL_CHAPTERS: ChapterEntry[] = PARTS.flatMap((p) => p.chapters);

export const chapterBySlug = (slug: string) => ALL_CHAPTERS.find((c) => c.slug === slug);
export const chapterByNumber = (n: number) => ALL_CHAPTERS.find((c) => c.n === n);

/** Part label for a chapter number, e.g. "PART II". */
export function partOf(n: number): PartEntry | undefined {
  return PARTS.find((p) => p.chapters.some((c) => c.n === n));
}
