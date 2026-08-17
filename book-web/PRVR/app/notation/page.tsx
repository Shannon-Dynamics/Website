import type { Metadata } from 'next';
import { NotationTable } from '@/components/book/notation-table';
import { ColorKey } from '@/components/book/color-key';
import { SiteHeader } from '@/components/shannon/site-header';
import { PageBanner, bookCrumb } from '@/components/shannon/page-banner';
import { BOOK_LINKS } from '@/lib/nav';

export const metadata: Metadata = {
  title: 'Notation',
  description:
    'Every symbol used in Probabilistic Robotics via Rust, following Thrun, Burgard and Fox, extended with the operators modern estimation needs on manifolds.',
};

export default function NotationPage() {
  return (
    <>
      <SiteHeader bookLinks={BOOK_LINKS} overlay />
      <PageBanner
        small
        crumb={bookCrumb(
          { label: 'PROBABILISTIC ROBOTICS VIA RUST', href: '/' },
          { label: 'NOTATION' },
        )}
        title="Notation"
        sub="Every symbol the book uses, in the notation lineage of Thrun, Burgard and Fox — extended where modern estimation needed it."
      />

      <main className="mx-auto w-full max-w-3xl px-4 py-14 sm:px-6">
        <p className="font-prose text-base leading-relaxed text-fd-muted-foreground">
          The book follows the notation of Thrun, Burgard and Fox so that readers can move between
          this text and the literature without translating. Where the field has moved on — chiefly
          estimation on manifolds — the notation is extended rather than replaced.
        </p>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold">The estimation loop</h2>
        <NotationTable
          rows={[
            { sym: 'x_t', meaning: 'State at time t — usually the robot pose, sometimes pose plus map' },
            { sym: 'u_t', meaning: 'Control (or odometry reading) applied between t-1 and t' },
            { sym: 'z_t', meaning: 'Measurement received at time t' },
            { sym: 'm', meaning: 'The map' },
            { sym: 'x_{0:t}', meaning: 'The trajectory: every state from 0 through t' },
            {
              sym: '\\bel(x_t) = p(x_t \\mid z_{1:t}, u_{1:t})',
              meaning: 'Belief: the posterior over the state given everything observed so far',
            },
            {
              sym: '\\belbar(x_t)',
              meaning: 'Predicted belief, after the control but before the measurement',
            },
            { sym: '\\eta', meaning: 'A normalizer, written generically wherever one is needed' },
            {
              sym: 'p(x_t \\mid x_{t-1}, u_t)',
              meaning: 'Motion model — the probability of landing in a state, given where you were and what you commanded',
            },
            {
              sym: 'p(z_t \\mid x_t)',
              meaning: 'Measurement model — the probability of a reading, given the state that produced it',
            },
          ]}
        />
      </section>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold">Gaussians</h2>
        <NotationTable
          rows={[
            { sym: '\\mu_t, \\Sigma_t', meaning: 'Moments form: mean and covariance' },
            {
              sym: '\\Omega_t = \\Sigma_t^{-1}, \\; \\xi_t = \\Sigma_t^{-1}\\mu_t',
              meaning: 'Canonical (information) form: information matrix and information vector',
              note: 'Cheap where moments form is expensive, and vice versa — Chapter 6',
            },
            { sym: 'R_t', meaning: 'Motion noise covariance', note: 'R for motion, always' },
            { sym: 'Q_t', meaning: 'Measurement noise covariance', note: 'Q for measurement, always' },
            { sym: 'G_t', meaning: 'Jacobian of the motion model with respect to the state' },
            { sym: 'H_t', meaning: 'Jacobian of the measurement model with respect to the state' },
            { sym: 'K_t', meaning: 'Kalman gain' },
            { sym: 'S_t', meaning: 'Innovation covariance' },
          ]}
        />
      </section>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold">Nonparametric representations</h2>
        <NotationTable
          rows={[
            {
              sym: '\\mathcal{X}_t = \\{ x_t^{[i]}, w_t^{[i]} \\}_{i=1}^{M}',
              meaning: 'A weighted particle set of M hypotheses',
            },
            { sym: 'M_{\\text{eff}}', meaning: 'Effective sample size — how many particles are really carrying the belief' },
            {
              sym: '\\ell_{t,i}',
              meaning: 'Log odds of occupancy for grid cell i',
              note: 'Log odds turns the Bayes update into addition — Chapter 13',
            },
          ]}
        />
      </section>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold">Geometry and manifolds</h2>
        <NotationTable
          rows={[
            { sym: 'T \\in \\SEtwo', meaning: 'A planar pose, as an element of the special Euclidean group' },
            { sym: '\\exp(\\tau), \\; \\log(T)', meaning: 'Exponential and logarithm maps between the tangent space and the group' },
            {
              sym: 'x \\bplus \\tau, \\; a \\bminus b',
              meaning: 'Retraction and its inverse: how you add a small increment to a pose, and subtract two poses',
              note: 'The operators that make Gaussian filters work on rotations — Chapters 3 and 7',
            },
            { sym: '\\Ad_T', meaning: 'Adjoint: moves a tangent vector between the left and right conventions' },
            { sym: '\\alpha_1 \\ldots \\alpha_6', meaning: 'Motion model noise parameters' },
            { sym: 'c_t', meaning: 'Correspondence variable: which landmark produced which measurement' },
            { sym: 'f_t^i = (r, \\phi, s)\\T', meaning: 'A detected feature: range, bearing, signature' },
          ]}
        />
      </section>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold">Decision making</h2>
        <NotationTable
          rows={[
            { sym: '\\pi', meaning: 'Policy: a mapping from state (or belief) to action' },
            { sym: 'r(x, u)', meaning: 'Immediate reward for taking action u in state x' },
            { sym: '\\gamma', meaning: 'Discount factor' },
            { sym: 'V(x)', meaning: 'Value function' },
            { sym: 'b', meaning: 'Belief state — the POMDP’s notion of what is known' },
          ]}
        />
      </section>

      <section className="mt-10">
        <h2 className="font-display text-xl font-semibold">Color</h2>
        <p className="mt-2 font-prose text-base leading-relaxed text-fd-muted-foreground">
          Color carries meaning in this book, consistently across prose, equations, figures, and
          code comments. A term tinted blue in an equation is the same quantity as the blue curve
          in the figure beside it.
        </p>
          <div className="mt-4">
            <ColorKey />
          </div>
        </section>
      </main>
    </>
  );
}
