import { Halftone } from './halftone';
import {
  LIBRARY_LINKS,
  SHANNON_CAPABILITIES,
  SHANNON_CONTACT,
  SHANNON_ECOSYSTEM,
  SHANNON_EMAIL,
  SHANNON_HOME,
  SHANNON_PRODUCTS,
  asset,
} from '@/lib/shannon';

/**
 * The Shannon Dynamics footer.
 *
 * The top bar is the book's own navigation; the walk back to the company lives
 * down here instead. The marketing site puts a contact form in this slot — the
 * book sends the reader to that one form rather than standing up a second
 * endpoint for the same enquiry.
 */
export function SiteFooter() {
  return (
    <footer className="sd-foot sd-tex sd-tex-foot">
      <Halftone speed={1.4} lens={false} />

      <div className="sd-foot-sec">
        <div className="sd-foot-in sd-foot-cta">
          <p className="sd-foot-eyebrow">Contact</p>
          <h2>
            Have a challenge?
            <br />
            Let&rsquo;s build what&rsquo;s next.
          </h2>
          <p>
            Whether you&rsquo;re exploring an intelligent machine, developing a new digital product,
            or solving a complex engineering problem — tell us what you&rsquo;re working on.
          </p>
          <a className="sd-foot-btn" href={SHANNON_CONTACT}>
            Start a conversation <span aria-hidden="true">→</span>
          </a>
        </div>
      </div>

      <div className="sd-foot-sec">
        <div className="sd-foot-in sd-foot-cols">
          <div className="sd-foot-brand">
            <a href={SHANNON_HOME}>
              <img src={asset('shannon/logo-horizontal-white.png')} alt="Shannon Dynamics" />
            </a>
            <p>Simulation-first engineering for robots and autonomous software.</p>
            <a className="sd-foot-mail" href={`mailto:${SHANNON_EMAIL}`}>
              {SHANNON_EMAIL}
            </a>
          </div>

          <div>
            <p className="sd-foot-col-head">Company</p>
            <ul className="sd-foot-col-list">
              <li>
                <a href={SHANNON_CAPABILITIES}>Capabilities</a>
              </li>
              <li>
                <a href={SHANNON_PRODUCTS}>Products</a>
              </li>
              <li>
                <a href={SHANNON_ECOSYSTEM}>Ecosystem</a>
              </li>
              <li>
                <a href={SHANNON_CONTACT}>Contact</a>
              </li>
            </ul>
          </div>

          <div>
            <p className="sd-foot-col-head">Library</p>
            <ul className="sd-foot-col-list">
              {LIBRARY_LINKS.map((item) => (
                <li key={item.href}>
                  <a href={item.href}>{item.title}</a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <p className="sd-foot-col-head">Connect</p>
            <ul className="sd-foot-col-list">
              <li>
                <a href="https://bohrlabs.id" target="_blank" rel="noopener">
                  Bohr Labs ↗
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>

      <div className="sd-foot-sec">
        <div className="sd-foot-in sd-foot-bar">
          <span>© 2026 Shannon Dynamics · A Bohr Labs Company</span>
          <span className="sd-foot-stamp">SIMULATION-FIRST</span>
        </div>
      </div>
    </footer>
  );
}
