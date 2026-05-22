import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/beta-quick-start">
            Beta quick start
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/architecture/overview">
            Architecture
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Documentation for Lumen Hub, a unified multimodal inference gateway.">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <h3>Three-layer design</h3>
                <p>
                  daemon (transport) → service (routing) → models (inference).
                  Each layer has a single responsibility and depends only on
                  the layer below.
                </p>
              </div>
              <div className="col col--4">
                <h3>Dynamic batching</h3>
                <p>
                  Preprocessed tensor requests are merged automatically to
                  improve ONNX and GPU throughput.
                </p>
              </div>
              <div className="col col--4">
                <h3>Pluggable models</h3>
                <p>
                  Factory → Service → Pipeline → Task is the standard
                  integration path for CLIP, SigLIP, PP-OCR, InsightFace, and
                  BioCLIP.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
