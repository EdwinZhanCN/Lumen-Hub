import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docsSidebar: [
    "intro",
    {
      type: "category",
      label: "Architecture",
      items: [
        "architecture/overview",
        "architecture/request-lifecycle",
        "architecture/batching",
        "architecture/model-pattern",
      ],
    },
    {
      type: "category",
      label: "Models",
      items: ["models/clip", "models/siglip", "models/fastvlm", "models/ppocr"],
    },
    {
      type: "category",
      label: "Configuration",
      items: ["configuration/lumen-config", "configuration/batching-config"],
    },
    {
      type: "category",
      label: "Development",
      items: ["development/adding-a-model"],
    },
  ],
};

export default sidebars;
