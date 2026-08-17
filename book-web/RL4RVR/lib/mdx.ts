import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';

const CONTENT_DIR = path.join(process.cwd(), 'content', 'chapters');

export interface ChapterFrontmatter {
  title: string;
  chapter: number;
  quote?: { text: string; author: string; affiliation?: string; source?: string };
}

export interface LoadedChapter {
  frontmatter: ChapterFrontmatter;
  content: string;
}

export function chapterFilePath(n: number, slug: string): string {
  return path.join(CONTENT_DIR, `ch${String(n).padStart(2, '0')}-${slug}.mdx`);
}

export function chapterExists(n: number, slug: string): boolean {
  return fs.existsSync(chapterFilePath(n, slug));
}

export function loadChapter(n: number, slug: string): LoadedChapter | null {
  const file = chapterFilePath(n, slug);
  if (!fs.existsSync(file)) return null;
  const raw = fs.readFileSync(file, 'utf8');
  const { data, content } = matter(raw);
  return { frontmatter: data as ChapterFrontmatter, content };
}

/** Extract `## ` headings for the in-page table of contents rail. */
export function extractHeadings(content: string): Array<{ id: string; text: string; level: 2 | 3 }> {
  const out: Array<{ id: string; text: string; level: 2 | 3 }> = [];
  const lines = content.split('\n');
  let inFence = false;
  for (const line of lines) {
    if (line.trimStart().startsWith('```')) inFence = !inFence;
    if (inFence) continue;
    const m = /^(#{2,3})\s+(.+?)\s*$/.exec(line);
    if (!m) continue;
    const level = m[1].length as 2 | 3;
    // Strip inline markdown/MDX so the rail shows clean text.
    const text = m[2]
      .replace(/\*\*/g, '')
      .replace(/`/g, '')
      .replace(/\$([^$]+)\$/g, '$1')
      .trim();
    out.push({ id: slugifyHeading(text), text, level });
  }
  return out;
}

export function slugifyHeading(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-');
}
