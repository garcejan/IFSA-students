This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

This app is fully static and needs **no environment variables**, so deployment is a link plus a production deploy.

### Deploy with Bob

Open the project in Bob (Agent mode) and use this prompt:

```text
Deploy this app to vercel. If necessary, ask for any information you need
```

Bob checks for the Vercel CLI and a login, asks you for what is missing (account or team scope, project name, production or preview), then links and deploys. Expect these steps:

1. **Log in.** Bob starts `vercel login` and gives you a URL with a device code. Open it, sign in, and approve. Tell Bob when you are done.
2. **Choose scope and project name.** If you do not answer, Bob uses the CLI's default scope and a project name such as `ppth-diagnostics`.
3. **Link and deploy.** Bob runs the commands below and reports the production URL.

### Deploy manually

```bash
npm install -g vercel          # once; skip if `vercel --version` already works
vercel login                   # opens a device-code URL in the browser
vercel link --yes --project <project-name>
vercel deploy --prod --yes
```

Use `vercel deploy --yes` instead (no `--prod`) for a preview URL. To deploy under a specific team, add `--scope <team-slug>` to the `link` command.

Vercel detects Next.js automatically; no build settings are needed. Redeploy after any change with `vercel deploy --prod --yes`.

`vercel link` creates `.vercel/` and `.env.local` in this folder. Both are git-ignored; do not commit them.

### Embedding the chat after deployment

The chat area on the page is an empty container, `#wxo-chat-root`, with a placeholder shown until something is mounted into it. Once you add your chat embed, redeploy. If your watsonx Orchestrate tenant restricts allowed origins, add your Vercel domain to its allow list.

### Troubleshooting

- **`No existing credentials found`**: run `vercel login` again, or pass `--token <token>` (create one under Vercel → Account Settings → Tokens).
- **Login never completes**: the device code expires after a few minutes. Start `vercel login` again and approve the new code.
- **Deployed under the wrong account**: run `vercel link --scope <team-or-username>` and deploy again.

More details: [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) and the [Vercel CLI reference](https://vercel.com/docs/cli).
