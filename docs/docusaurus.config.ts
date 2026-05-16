import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "Lumen Hub",
  tagline: "Unified Multimodal Inference Gateway",
  favicon: "img/favicon.ico",

  future: {
    v4: true,
  },

  url: "https://lumen-rs.dev",
  baseUrl: "/hub/",

  organizationName: "lumen-rs",
  projectName: "lumen-rs",

  onBrokenLinks: "throw",

  i18n: {
    defaultLocale: "zh-Hans",
    locales: ["zh-Hans"],
  },

  markdown: {
    mermaid: true,
  },
  themes: ["@docusaurus/theme-mermaid"],

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          editUrl: "https://github.com/lumen-rs/lumen-rs/tree/main/docs/",
          routeBasePath: "/",
        },
        blog: false,
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: "img/docusaurus-social-card.jpg",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "Lumen Hub",
      logo: {
        alt: "Lumen Hub Logo",
        src: "img/logo.svg",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "docsSidebar",
          position: "left",
          label: "Docs",
        },
        {
          href: "https://github.com/lumen-rs/lumen-rs",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Architecture Overview",
              to: "/architecture/overview",
            },
            {
              label: "Batching Design",
              to: "/architecture/batching",
            },
          ],
        },
        {
          title: "Models",
          items: [
            { label: "CLIP", to: "/models/clip" },
            { label: "SigLIP", to: "/models/siglip" },
            { label: "FastVLM", to: "/models/fastvlm" },
            { label: "PP-OCR", to: "/models/ppocr" },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/lumen-rs/lumen-rs",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Lumen. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["rust", "toml", "json"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
