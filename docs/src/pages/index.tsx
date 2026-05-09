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
            to="/architecture/overview">
            阅读架构文档
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
      description="Lumen Hub 统一多模态推理网关文档">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className="col col--4">
                <h3>三层架构</h3>
                <p>daemon（传输）→ service（抽象）→ models（推理），职责清晰，逐层依赖。</p>
              </div>
              <div className="col col--4">
                <h3>动态批处理</h3>
                <p>对预处理张量请求自动合并批次，提升 GPU/ONNX 推理吞吐。</p>
              </div>
              <div className="col col--4">
                <h3>模型可插拔</h3>
                <p>Factory → Service → Pipeline → Task 统一集成模式，新模型可快速接入。</p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
