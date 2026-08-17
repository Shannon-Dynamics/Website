import { defineDocs, defineConfig, frontmatterSchema } from 'fumadocs-mdx/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import * as z from 'zod';
import { katexOptions } from './lib/katex-macros';

/**
 * Chapter frontmatter. Everything beyond title/description is book-specific
 * metadata that the chapter chrome renders (part label, difficulty, the
 * researcher epigraph, and the estimated reading time).
 */
export const docs = defineDocs({
  dir: 'content/chapters',
  docs: {
    schema: frontmatterSchema.extend({
      chapter: z.number().optional(),
      part: z.string().optional(),
      partTitle: z.string().optional(),
      difficulty: z.enum(['Foundational', 'Intermediate', 'Advanced']).optional(),
      readingTime: z.string().optional(),
      quote: z.string().optional(),
      quoteAuthor: z.string().optional(),
      quoteSource: z.string().optional(),
    }),
  },
});

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    rehypePlugins: (v) => [
      // The macro table must be a mutable copy: KaTeX writes \cr into it for
      // every matrix/cases/aligned environment. See lib/katex-macros.ts.
      [rehypeKatex, katexOptions()],
      ...v,
    ],
  },
});
