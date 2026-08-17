import defaultComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';

import { Callout } from 'fumadocs-ui/components/callout';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import { Accordion, Accordions } from 'fumadocs-ui/components/accordion';

import { Overview } from '@/components/book/overview';
import { Epigraph } from '@/components/book/epigraph';
import { Derivation } from '@/components/book/derivation';
import { Algorithm } from '@/components/book/algorithm';
import { Exercises, Exercise } from '@/components/book/exercises';
import { References, Reference } from '@/components/book/references';
import { NotationTable } from '@/components/book/notation-table';
import { KeyIdea } from '@/components/book/key-idea';
import { Figure } from '@/components/book/figure';
import { WidgetFrame } from '@/components/sim/widget-frame';
import { ColorKey } from '@/components/book/color-key';
import { Scrub } from '@/components/book/scrub';
import { LinkedMath, RoleTag } from '@/components/book/linked-math';
import { Predict, CheckAnswer, Hints } from '@/components/book/predict';

/** Chapter-authoring surface: every component an MDX chapter may use. */
export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultComponents,
    Callout,
    Tab,
    Tabs,
    Step,
    Steps,
    Accordion,
    Accordions,
    Overview,
    Epigraph,
    Derivation,
    Algorithm,
    Exercises,
    Exercise,
    References,
    Reference,
    NotationTable,
    KeyIdea,
    Figure,
    WidgetFrame,
    ColorKey,
    Scrub,
    LinkedMath,
    RoleTag,
    Predict,
    CheckAnswer,
    Hints,
    ...components,
  };
}

export const useMDXComponents = getMDXComponents;
