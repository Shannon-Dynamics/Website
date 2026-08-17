import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import rehypePrettyCode from 'rehype-pretty-code';

import { Callout, Proof, Theorem } from './Callout';
import { ResearcherQuote } from './ResearcherQuote';
import { Pre, RustSnippet } from './CodeBlock';
import { CodingTask, Exercises } from './Exercises';
import { References } from './References';
import { ChapterOverview } from './ChapterOverview';
import { Figure } from './Figure';

// Interactive simulations available to every chapter's MDX.
import { RustyDrive } from '@/components/sim/RustyDrive';
import { SuccessLevels } from '@/components/sim/SuccessLevels';
import { PendleSim } from '@/components/sim/PendleSim';
import { BanditTestbed } from '@/components/sim/BanditTestbed';
import { GpiDashboard } from '@/components/sim/GpiDashboard';
import { MdpExplorer } from '@/components/sim/MdpExplorer';
import { TdDashboard } from '@/components/sim/TdDashboard';
import { LambdaDial } from '@/components/sim/LambdaDial';
import { ContractionDemo } from '@/components/sim/ContractionDemo';
import { CurseOfDimensionality } from '@/components/sim/CurseOfDimensionality';
import { DomainRandomization } from '@/components/sim/DomainRandomization';
import { CovariateShift } from '@/components/sim/CovariateShift';
import { PolicyGradientLab } from '@/components/sim/PolicyGradientLab';
import { RewardMixer } from '@/components/sim/RewardMixer';
import { DeadlyTriad } from '@/components/sim/DeadlyTriad';
import { ReplayBuffer } from '@/components/sim/ReplayBuffer';
import { ModelBiasFan } from '@/components/sim/ModelBiasFan';
import { EntropyDial } from '@/components/sim/EntropyDial';
import { DmpSculptor } from '@/components/sim/DmpSculptor';
import { PipelineSwitcher } from '@/components/sim/PipelineSwitcher';
import { GraspWrench } from '@/components/sim/GraspWrench';
import { ReacherKinematics } from '@/components/sim/ReacherKinematics';
import { SharedAutonomy } from '@/components/sim/SharedAutonomy';
import { MissionControl } from '@/components/sim/MissionControl';
import { WarehouseEditor } from '@/components/sim/WarehouseEditor';
import { RewardDesigner } from '@/components/sim/RewardDesigner';
import { LineChart } from '@/components/viz/LineChart';
import { BarChart } from '@/components/viz/BarChart';
import { StatRow, StatTile } from '@/components/viz/StatTile';

const components = {
  // Prose primitives
  pre: Pre,
  Figure,

  // Book furniture
  ChapterOverview,
  ResearcherQuote,
  Callout,
  Theorem,
  Proof,
  RustSnippet,
  Exercises,
  CodingTask,
  References,

  // Charts
  LineChart,
  BarChart,
  StatRow,
  StatTile,

  // Simulations
  RustyDrive,
  SuccessLevels,
  PendleSim,
  BanditTestbed,
  GpiDashboard,
  MdpExplorer,
  TdDashboard,
  LambdaDial,
  ContractionDemo,
  CurseOfDimensionality,
  DomainRandomization,
  CovariateShift,
  PolicyGradientLab,
  RewardMixer,
  DeadlyTriad,
  ReplayBuffer,
  ModelBiasFan,
  EntropyDial,
  DmpSculptor,
  PipelineSwitcher,
  GraspWrench,
  ReacherKinematics,
  SharedAutonomy,
  MissionControl,
  WarehouseEditor,
  RewardDesigner,
};

const prettyCodeOptions = {
  theme: { light: 'github-light', dark: 'github-dark-dimmed' },
  keepBackground: false,
  defaultLang: 'rust',
};

export function MdxContent({ source }: { source: string }) {
  return (
    <MDXRemote
      source={source}
      components={components}
      options={{
        mdxOptions: {
          remarkPlugins: [remarkGfm, remarkMath],
          rehypePlugins: [
            rehypeSlug,
            [rehypeKatex, { strict: false, throwOnError: false }],
            [rehypePrettyCode, prettyCodeOptions],
          ],
        },
      }}
    />
  );
}
