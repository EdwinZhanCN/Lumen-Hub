import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: '架构设计',
      items: [
        'architecture/overview',
        'architecture/request-lifecycle',
        'architecture/batching',
        'architecture/model-pattern',
      ],
    },
    {
      type: 'category',
      label: '模型',
      items: [
        'models/clip',
        'models/siglip',
        'models/fastvlm',
      ],
    },
    {
      type: 'category',
      label: '配置',
      items: [
        'configuration/lumen-config',
        'configuration/batching-config',
      ],
    },
    {
      type: 'category',
      label: '开发指南',
      items: [
        'development/adding-a-model',
      ],
    },
  ],
};

export default sidebars;
