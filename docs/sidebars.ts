import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebars: SidebarsConfig = {
  docsSidebar: [
    "intro",
    "beta-quick-start",
    {
      type: "category",
      label: "Architecture",
      items: [
        "architecture/overview",
        "architecture/request-lifecycle",
        "architecture/batching",
        "architecture/model-pattern",
        "architecture/task-input",
        "architecture/task-request-examples",
      ],
    },
    {
      type: "category",
      label: "Models",
      items: [
        "models/clip",
        "models/bioclip",
        "models/insightface",
        "models/siglip",
        "models/ppocr",
      ],
    },
    {
      type: "category",
      label: "Configuration",
      items: ["configuration/lumen-config", "configuration/batching-config"],
    },
    {
      type: "category",
      label: "Development",
      items: [
        "development/adding-a-model",
        "development/beta-local-dist",
        "development/model-repository",
        "development/openvino-package",
      ],
    },
  ],
};

export default sidebars;
