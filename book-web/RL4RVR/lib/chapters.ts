/**
 * The book's table of contents — mirrors TOC.md exactly.
 * TOC.md remains the single source of truth; this file is its typed shadow.
 */

export type PartId = 'I' | 'II' | 'III' | 'IV' | 'V';

export interface Part {
  id: PartId;
  title: string;
  tagline: string;
  chapters: number[];
}

export interface ChapterMeta {
  n: number;
  slug: string;
  title: string;
  part: PartId;
  /** One-line hook shown on cards and in the sidebar. */
  blurb: string;
  /** Baseline references this chapter modernizes. */
  sources: string[];
  /** Cast robots that appear. */
  robots: string[];
  /** Whether the MDX content has been written yet. */
  published: boolean;
}

export const PARTS: Part[] = [
  {
    id: 'I',
    title: 'Foundations of Sequential Decision-Making',
    tagline: "Sutton & Barto's spine, retold with robots and interactive math.",
    chapters: [1, 2, 3, 4, 5, 6, 7],
  },
  {
    id: 'II',
    title: 'Scaling Up: Function Approximation & Deep RL',
    tagline: 'From tables to tensors — the leap robots require.',
    chapters: [8, 9, 10, 11, 12],
  },
  {
    id: 'III',
    title: 'The Robotics Side',
    tagline: "Kober's bridge, rebuilt with modern materials.",
    chapters: [13, 14, 15, 16, 17],
  },
  {
    id: 'IV',
    title: 'Competencies: RL on Real Robots',
    tagline: "Tang's taxonomy as deep dives: what worked, why, and rebuilt in Rust.",
    chapters: [18, 19, 20],
  },
  {
    id: 'V',
    title: 'Frontiers & Capstone',
    tagline: 'Where the field is going — and one project that uses all of it.',
    chapters: [21, 22],
  },
];

export const CHAPTERS: ChapterMeta[] = [
  {
    n: 1,
    slug: 'why-rl-for-robotics',
    title: 'Why Reinforcement Learning for Robotics?',
    part: 'I',
    blurb:
      'The see–think–act loop, why hand-coding breaks, and a tour of real-world successes graded L0–L5.',
    sources: ['Kober §1', 'Tang §1–3', 'Akinola'],
    robots: ['Rusty'],
    published: true,
  },
  {
    n: 2,
    slug: 'mathematical-toolkit',
    title: 'The Mathematical Toolkit',
    part: 'I',
    blurb:
      'Probability, linear algebra, ODE discretization, contraction mappings and the Robbins–Monro theorem that makes every RL algorithm tick.',
    sources: ['S&B prerequisites', 'Kober §1.3'],
    robots: ['Pendle'],
    published: true,
  },
  {
    n: 3,
    slug: 'multi-armed-bandits',
    title: 'Multi-Armed Bandits: Exploration & Exploitation',
    part: 'I',
    blurb:
      'The atom of RL: action values, regret, UCB derived from Hoeffding, gradient bandits as the seed of policy gradients.',
    sources: ['S&B ch. 2'],
    robots: ['Reacher'],
    published: true,
  },
  {
    n: 4,
    slug: 'markov-decision-processes',
    title: 'Markov Decision Processes: The Formalism',
    part: 'I',
    blurb:
      'The MDP tuple, Bellman equations derived in full, γ-contraction proofs, and POMDPs — because robots never see state.',
    sources: ['S&B ch. 3', 'Tang §3.2'],
    robots: ['Rusty'],
    published: true,
  },
  {
    n: 5,
    slug: 'dynamic-programming',
    title: 'Dynamic Programming: Planning with a Known Model',
    part: 'I',
    blurb:
      "Policy evaluation, the policy improvement theorem, value iteration, and generalized policy iteration — the book's master pattern.",
    sources: ['S&B ch. 4'],
    robots: ['Rusty'],
    published: true,
  },
  {
    n: 6,
    slug: 'monte-carlo-and-td',
    title: 'Learning from Experience: Monte Carlo & Temporal-Difference',
    part: 'I',
    blurb:
      'Learning without a model: MC, importance sampling, TD(0), SARSA vs Q-learning, and the maximization bias that Double Q fixes.',
    sources: ['S&B ch. 5–6'],
    robots: ['Rusty'],
    published: true,
  },
  {
    n: 7,
    slug: 'traces-planning-mcts',
    title: 'Unifying Learning & Planning: n-step, Traces, Dyna & MCTS',
    part: 'I',
    blurb:
      'The λ-dial between TD and MC, eligibility traces, Dyna-Q learning from imagination, and MCTS at decision time.',
    sources: ['S&B ch. 7–8, 12'],
    robots: ['Rusty'],
    published: true,
  },
  {
    n: 8,
    slug: 'function-approximation',
    title: 'Function Approximation & the Deadly Triad',
    part: 'II',
    blurb:
      'Why robots outgrow tables: semi-gradient TD, tile coding, and Baird’s counterexample diverging live.',
    sources: ['S&B ch. 9–11', 'Kober §4.2'],
    robots: ['Pendle'],
    published: true,
  },
  {
    n: 9,
    slug: 'deep-value-methods',
    title: 'Deep Value-Based Methods: DQN & Descendants',
    part: 'II',
    blurb:
      'Replay buffers and target networks as variance surgery; Double, Dueling, PER, Rainbow, and distributional RL.',
    sources: ['Mnih 2015', 'Tang Table 5'],
    robots: ['Rusty'],
    published: true,
  },
  {
    n: 10,
    slug: 'policy-gradients',
    title: 'Policy Gradients: REINFORCE → PPO',
    part: 'II',
    blurb:
      'The policy gradient theorem derived step by step, baselines and GAE, trust regions, and PPO — the workhorse of real robot RL.',
    sources: ['S&B ch. 13', 'Kober §2.2.2', 'Schulman 2017'],
    robots: ['Pendle'],
    published: true,
  },
  {
    n: 11,
    slug: 'off-policy-continuous-control',
    title: 'Off-Policy Continuous Control: DDPG, TD3 & SAC',
    part: 'II',
    blurb:
      'Sample efficiency as a robotics imperative: the DPG theorem, overestimation bias, and maximum-entropy RL.',
    sources: ['Silver 2014', 'Fujimoto 2018', 'Haarnoja 2018'],
    robots: ['Reacher'],
    published: true,
  },
  {
    n: 12,
    slug: 'model-based-rl',
    title: 'Model-Based RL & World Models',
    part: 'II',
    blurb:
      'Mental rehearsal: ensemble dynamics, model-bias compounding bounds, CEM-MPC, and learning inside a latent dream.',
    sources: ['Kober §6', 'Chua 2018', 'Hafner 2023'],
    robots: ['Pendle'],
    published: true,
  },
  {
    n: 13,
    slug: 'robot-as-environment',
    title: 'The Robot as an Environment: Kinematics, Dynamics & Control',
    part: 'III',
    blurb:
      'What is inside the env black box when it is a robot: FK/IK, Jacobians, the manipulator equation, PID and LQR baselines.',
    sources: ['Kober §1.3', 'Spong 2006', 'Siciliano 2009'],
    robots: ['Reacher', 'Pendle'],
    published: true,
  },
  {
    n: 14,
    slug: 'four-curses',
    title: 'The Four Curses of Robot RL',
    part: 'III',
    blurb:
      'Dimensionality, real-world samples, under-modeling and goal specification — each quantified, plus the reward-hacking zoo.',
    sources: ['Kober §3', 'Tang §5', 'Ng 1999'],
    robots: ['Reacher'],
    published: true,
  },
  {
    n: 15,
    slug: 'sim-to-real',
    title: 'Simulation & the Sim-to-Real Bridge',
    part: 'III',
    blurb:
      'Integrators, contact models, domain randomization as distributional robustness, system identification and teacher–student transfer.',
    sources: ['Kober §6.1', 'Tang §3.3'],
    robots: ['Ferris', 'Pendle'],
    published: true,
  },
  {
    n: 16,
    slug: 'imitation-and-offline-rl',
    title: 'Demonstrations, Imitation & Offline RL',
    part: 'III',
    blurb:
      'Behavior cloning and its T² compounding error, DAgger, MaxEnt IRL, CQL/IQL pessimism, and offline-to-online fine-tuning.',
    sources: ['Kober §5', 'Akinola LfD', 'Ross 2011', 'Kostrikov 2021'],
    robots: ['Reacher', 'Rusty'],
    published: true,
  },
  {
    n: 17,
    slug: 'motor-skill-representations',
    title: 'Motor-Skill Policy Representations',
    part: 'III',
    blurb:
      'What the policy outputs matters as much as how it learns: action-space levels, DMPs, CPGs, residual RL, and ball-in-a-cup then vs now.',
    sources: ['Kober §4, §7', 'Ijspeert 2013', 'Tang §3.2'],
    robots: ['Reacher'],
    published: true,
  },
  {
    n: 18,
    slug: 'learning-locomotion',
    title: 'Learning Locomotion',
    part: 'IV',
    blurb:
      'DRL’s flagship competency: reward anatomy, terrain curricula, teacher–student privileged learning, and Ferris learning to walk.',
    sources: ['Tang §4.1', 'Lee 2020', 'Kumar 2021'],
    robots: ['Ferris'],
    published: true,
  },
  {
    n: 19,
    slug: 'navigation-and-mobile-manipulation',
    title: 'Learning Navigation & Mobile Manipulation',
    part: 'IV',
    blurb:
      'Belief MDPs and recurrent policies, end-to-end vs modular pipelines, hierarchical skills, and whole-body control.',
    sources: ['Tang §4.2', 'Tang §4.4'],
    robots: ['Rusty', 'Reacher'],
    published: true,
  },
  {
    n: 20,
    slug: 'learning-manipulation',
    title: 'Learning Manipulation',
    part: 'IV',
    blurb:
      'The hard competency: grasp wrench space, impedance action spaces, in-hand dexterity, and honest success-rate methodology.',
    sources: ['Tang §4.3', 'OpenAI 2020'],
    robots: ['Reacher'],
    published: true,
  },
  {
    n: 21,
    slug: 'frontiers',
    title: 'Frontiers: HRI, Multi-Robot & Foundation Models',
    part: 'V',
    blurb:
      'Shared autonomy, Dec-POMDPs and CTDE, vision-language-action models, and a Kober-2013-vs-Tang-2024 open-problem diff.',
    sources: ['Tang §4.5–4.6, §5', 'Kober §8'],
    robots: ['Rusty', 'Reacher'],
    published: true,
  },
  {
    n: 22,
    slug: 'capstone',
    title: 'Capstone: An End-to-End Learned Robot in Rust',
    part: 'V',
    blurb:
      'One project using the whole book: specify, formalize, randomize, train teacher–student PPO, evaluate honestly, and ship to the browser.',
    sources: ['Kober §7 discipline', 'Tang §3.4 rubric'],
    robots: ['Ferris'],
    published: true,
  },
];

export const CHAPTER_BY_SLUG = new Map(CHAPTERS.map((c) => [c.slug, c]));
export const CHAPTER_BY_NUMBER = new Map(CHAPTERS.map((c) => [c.n, c]));

export function getChapter(slug: string): ChapterMeta | undefined {
  return CHAPTER_BY_SLUG.get(slug);
}

export function getPart(id: PartId): Part {
  const part = PARTS.find((p) => p.id === id);
  if (!part) throw new Error(`unknown part ${id}`);
  return part;
}

export function neighbours(n: number): { prev?: ChapterMeta; next?: ChapterMeta } {
  return { prev: CHAPTER_BY_NUMBER.get(n - 1), next: CHAPTER_BY_NUMBER.get(n + 1) };
}

/** Cast robots, fixed vocabulary across the whole book. */
export const ROBOTS = {
  Rusty: {
    name: 'Rusty',
    kind: 'differential-drive mobile robot',
    intro: 1,
    thread: 'hello-robot → gridworld → visual gridworld → lidar navigation → multi-agent',
  },
  Pendle: {
    name: 'Pendle',
    kind: 'pendulum / cart-pole',
    intro: 2,
    thread: 'classical control ↔ RL bridge',
  },
  Reacher: {
    name: 'Reacher',
    kind: '2-link planar arm',
    intro: 3,
    thread: 'manipulation thread',
  },
  Ferris: {
    name: 'Ferris',
    kind: 'quadruped',
    intro: 15,
    thread: 'locomotion & sim-to-real thread',
  },
} as const;
