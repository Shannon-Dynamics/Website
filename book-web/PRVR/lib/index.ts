/**
 * Public surface of the book's algorithm library.
 *
 * Everything a chapter widget needs, in dependency order: randomness and linear
 * algebra at the bottom, then geometry, then the world, then the models that
 * relate poses to measurements, then the filters that invert them, then
 * mapping. Import from `@/lib` for the whole surface, or from the individual
 * module when a widget only needs one piece.
 */

// Foundations
export * from './prob/rng';
export * from './prob/linalg';
export * from './prob/gaussian';

// Geometry
export * from './geom/se2';

// Simulated worlds
export * from './sim/world';

// Motion and measurement models (Thrun Ch. 5–6)
export * from './models/motion';
export * from './models/sensor';

// The Bayes filter family (Thrun Ch. 2–4, 8)
export * from './filters/bayes';
export * from './filters/kf';
export * from './filters/ekf';
export * from './filters/ukf';
export * from './filters/pf';

// Mapping (Thrun Ch. 9)
export * from './mapping/occgrid';

// Numerical self-verification
export * from './__checks__';
