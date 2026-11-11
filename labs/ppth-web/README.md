# Build the Princeton-Plainsboro landing page with Bob

A step-by-step tutorial. You will prompt **IBM Bob** to build a futuristic, scroll-driven hospital web page for the diagnostics department you built in the watsonx Orchestrate lab (Dr House and his team), run it locally, understand how it works, and leave a ready-made spot for the embedded chat.

Deploying to Vercel is **optional** (Step 8).

| | |
|---|---|
| **Time** | About 30 minutes (plus 10 for the optional deploy) |
| **Result** | A Next.js page with scrollytelling, a routing animation, patient charts and a reserved chat container |
| **Stack** | [Next.js](https://nextjs.org) (App Router) · [Tailwind CSS](https://tailwindcss.com) v4 · [Motion](https://motion.dev) (`motion/react`) · IBM Plex fonts |

## What the department looks like

The page tells the story of the lab's multi-agent system: **7 agents, 6 tools, 2 skills**.

| Agent | Role | Gets cases about |
|---|---|---|
| `dr_house` | Orchestrator | Everything first: pulls the chart, then routes |
| `dr_foreman` | Neurology | Seizure, aphasia, paralysis, stroke |
| `dr_wilson` | Oncology | Masses, night sweats, lymphoma, LDH |
| `dr_chase` | Cardiology / ICU | Arrhythmia, shock, troponin, INR |
| `dr_cameron` | Immunology / infectious disease | Fever, transplant, any drug question |
| `dr_cuddy` | Administration | Consent, lawsuits, budget, beds |
| `case_reviewer` | Quality observer | Grades an existing answer |

The medicine is synthetic and deliberately shallow; the orchestration is real. All patients are fictional.

## Prerequisites

- **IBM Bob** open on your laptop, in **Agent** mode
- **Node.js 20 or newer** and **npm** (check with `node -v` and `npm -v`)
- The lab material (agents, tools and patient charts) in your workspace. Bob reads it to fill the page with real content, such as agent descriptions, patient IDs and tool names.
- Internet access during the first build, because the IBM Plex fonts are fetched from Google Fonts

You do **not** need a watsonx Orchestrate instance for this tutorial. The chat is embedded later.

---

## (Optional) Step 1: Describe what you want to build

Open your lab workspace in Bob and send a prompt that contains four things: the **scenario**, the **audience**, the **page requirements**, and the **stack**. This is the prompt used to create this project:

```text
This is the description of what I will be building: The Princeton-Plainsboro
diagnostics department — Dr House and his team, as a live multi-agent system on
watsonx Orchestrate. Seven agents, six tools, two agent skills. Dr House takes a
case, pulls the chart, and calls the colleague who owns the finding — that
delegation is the thing you will watch, measure and debug.

Users will access this chat as a watsonx Orchestrate embedded chat. I want you to
create a web page where this chat will be available as an embedded element.

Webpage requirements:
- fitting the style of a hospital page
- framework: Next.js
- Component UI Motion: Motion (motion/react)
- Styling: Tailwind CSS
- achieve a futuristic, interactive "scrollytelling" experience (similar to sites
  featured on Awwwards or Apple product showcases)
```

If you have an architecture diagram of the agents, attach it. Bob reads the image and uses the agent names and relationships.

> **Tip:** Name the framework, the animation library and the styling approach explicitly. Without them Bob will choose for you.


## Step 2: Run it locally

If you built the page yourself with Bob, the project folder already exists. If you are starting from this repository, install dependencies first:

```bash
cd labs/ppth-web
npm install        # only needed once
npm run dev
```

Open **http://localhost:3000**.

If port 3000 is taken, run `npm run dev -- -p 3111`.

## Step 3: Take the tour

Each section uses a different scroll technique. Read the component named in the table to see how it works.

| # | Section | What you see | Technique | File |
|---|---|---|---|---|
| 1 | Hero | Headline, live counters, ECG | Staggered text reveal, parallax with `useScroll` / `useTransform`, count-up with `animate` | `src/components/Hero.tsx` |
| 2 | Mission | Words fade in as you scroll | Per-word opacity tied to scroll progress | `src/components/Manifesto.tsx` |
| 3 | Dispatch | Four steps while the screen stays pinned | `position: sticky` in a tall section, step chosen from scroll progress | `src/components/Dispatch.tsx` |
| 4 | Team | Cards move sideways | Vertical scroll mapped to horizontal `x` | `src/components/Team.tsx` |
| 5 | Routing lab | Click a case, watch it route | State machine (phases 1–4) driving the SVG graph | `src/components/RouterLab.tsx` |
| 6 | Charts | Four patient cards | 3D tilt from mouse position with `useSpring` | `src/components/Charts.tsx` |
| 7 | Toolbelt | Six tools, two skills | `whileInView` with staggered children | `src/components/Toolbelt.tsx` |
| 8 | Consult | Chat panel and starter prompts | The embed point (Step 7) | `src/components/Consult.tsx` |

Shared pieces:

- `src/components/RoutingGraph.tsx`: the animated SVG used by both the Dispatch section and the routing lab
- `src/lib/data.ts`: **all content** (agents, patients, sample cases, tools). Edit text here, not in the components.
- `src/components/MotionProvider.tsx`: switches motion off for visitors who set "reduce motion" in their operating system

## (Optional) Step 4: Change it by prompting

You can modify your page now. Good follow-up prompts are small and specific. Try these:

```text
Make the colour palette warmer: replace the teal accent with a deep blue.
```

```text
Add a new section after the Team section that shows a timeline of one case
from arrival to answer, scrolling horizontally.
```

```text
The routing lab only has six cases. Add two more, one for Dr Cameron's
formulary check and one for a question that needs both Dr Chase and Dr Foreman.
```

```text
Make the hero headline smaller on mobile and check that the page has no
horizontal scroll at 390 px width.
```

> **Tip:** Ask Bob to run `npm run lint` and `npm run build` after each change. A change is finished when both pass.

## Step 6: Validate

Before you move on, run the same checks Bob ran:

```bash
npm run lint       # no errors expected
npm run build      # must finish with "Compiled successfully"
npm start          # serves the production build on http://localhost:3000
```

Production mode (`npm start`) is also the smoothest way to judge the scroll animations.

## Step 7: Connect the chat (when it is ready)

The Consult section contains this container, deliberately empty:

```html
<div id="wxo-chat-root" class="absolute inset-0"></div>
```

How it works:

- A placeholder ("Awaiting channel") sits behind it
- The moment your embed script adds an element **inside** `#wxo-chat-root`, a CSS rule hides the placeholder (`.chat-frame:has(#wxo-chat-root > *)` in `src/app/globals.css`)
- React renders the container with no children, so it never fights a third-party script

To connect it, generate the embed snippet for `dr_house` from your watsonx Orchestrate instance (the lab's part 3 shows how), set the snippet's `rootElementID` to `wxo-chat-root`, and load the script from the Consult component.

> **Note:** an embedded chat that works on a fully public page needs the web chat security settings for your agent and origin to be configured in the watsonx Orchestrate UI. Allow both `http://localhost:3000` for testing and, if you deploy, your public domain.

A prompt you can give Bob when you are ready:

```text
Mount the watsonx Orchestrate embedded chat into the #wxo-chat-root container
in the Consult section. Use this snippet: <paste your snippet>. Read the
config values from environment variables, not from the source code.
```

## Step 8 (Bonus): Deploy to Vercel

Skip this step if you only need the page locally. The app is fully static and needs **no environment variables**.

### Option A: let Bob do it

Use this prompt:

```text
Deploy this app to Vercel. If necessary, ask for any information you need.
```

Bob checks for the Vercel CLI and a login, asks for what is missing, then links and deploys. Expect:

1. **Log in.** Bob starts `vercel login` and gives you a URL with a device code. Open it, sign in, approve, and tell Bob you are done.
2. **Choose a scope and project name.** If you do not answer, Bob uses the CLI's default scope and a name such as `ppth-diagnostics`.
3. **Link and deploy.** Bob reports the production URL.

### Option B: do it yourself

```bash
npm install -g vercel          # once; skip if `vercel --version` works
vercel login                   # opens a device-code URL in the browser
vercel link --yes --project <project-name>
vercel deploy --prod --yes
```

- Use `vercel deploy --yes` (without `--prod`) for a preview URL
- To deploy under a team, add `--scope <team-slug>` to the `link` command
- Redeploy after any change with `vercel deploy --prod --yes`
- `vercel link` creates `.vercel/` and `.env.local` in this folder. Both are git-ignored; never commit them.
- Vercel detects Next.js automatically, so no build settings are needed

After you add the chat embed (Step 7), redeploy, and allow your Vercel domain in your watsonx Orchestrate tenant if it restricts origins.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `npm run dev` fails or modules are missing | Run `npm install` in `labs/ppth-web` |
| Build fails while fetching fonts | The build needs internet access for Google Fonts. Check your connection or proxy and retry. |
| Port 3000 is already in use | `npm run dev -- -p 3111` |
| Animations feel slow in dev mode | Use `npm run build && npm start` |
| Animations do not play | Your operating system has "reduce motion" enabled. This is intended. |
| Bob starts writing watsonx Orchestrate embed code you do not want | Send: "Do not embed watsonx Orchestrate. I will embed it myself later." |
| Counters in the hero show `0` in a screenshot | They count up after about 1.2 s. Wait, or take the screenshot later. |
| `vercel`: `No existing credentials found` | Run `vercel login`, or pass `--token <token>` (Vercel → Account Settings → Tokens) |
| `vercel login` never completes | The device code expires after a few minutes. Start `vercel login` again and approve the new code. |
| Deployed under the wrong account | `vercel link --scope <team-or-username>`, then deploy again |

## Project structure

```text
labs/ppth-web/
├── src/
│   ├── app/
│   │   ├── layout.tsx          # fonts (IBM Plex), metadata, MotionProvider
│   │   ├── page.tsx            # assembles the sections in order
│   │   └── globals.css         # Tailwind theme, graph-paper backgrounds, chat placeholder rule
│   ├── components/             # one file per section + RoutingGraph + MotionProvider
│   └── lib/data.ts             # all content: agents, patients, sample cases, tools, skills
├── next.config.ts
└── package.json
```

## Learn more

- [Next.js documentation](https://nextjs.org/docs) and [deployment guide](https://nextjs.org/docs/app/building-your-application/deploying)
- [Motion for React](https://motion.dev/docs/react): `useScroll`, `useTransform`, `whileInView`
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Vercel CLI reference](https://vercel.com/docs/cli)

*Synthetic teaching case. Not medical advice.*
