# IFSA workshop — Part 3 webchat embedding

## Embed Dr House into the Princeton-Plainsboro Next.js Web App

**~20 minutes.** In this lab, you will take the diagnostics multi-agent team (Dr House and his specialists) you deployed on watsonx Orchestrate and embed them into a beautiful, scroll-driven, futuristic Next.js web application (`ppth-web`). 

Instead of typing prompts into a terminal or CLI, you and your users will consult Dr. House through a highly responsive, custom-embedded web chat built right into the hospital’s diagnostic portal.

---

## Step 1 — Retrieve Credentials from watsonx Orchestrate UI

To embed the agent, you need five specific parameters from your deployed `dr_house` agent in the watsonx Orchestrate console.

### How to find them:

1. **Log in** to your IBM watsonx Orchestrate instance.
2. Navigate to the **Agents** tab or agents list and select your orchestrator agent, **`dr_house`**.
3. In the agent details view, click on the **Deploy** tab at the top.
4. From the left-hand menu under **Deploy** -> **Live**, click on **Embedded agent** (sometimes labeled *Embed agent* or *Web chat* channel).
5. Locate the **Embed code** section. This block contains an HTML snippet that looks like this:

```html
<script>
  window.wxOConfiguration = {
    orchestrationID: "*****",
    hostURL: "*****",
    rootElementID: "root",
    deploymentPlatform: "ibmcloud",
    crn: "crn:v1:bluemix:public:watsonx-orchestrate:*****",
    chatOptions: {
      agentId: "*****",
      agentEnvironmentId: "*****",
      onLoad: (instance) => {
        // ...
      }
    }
  };
</script>
```

### Identify your 5 key credentials:

- **`orchestrationID`**: The unique identifier linking your configuration to the instance.
- **`hostURL`**: The region-specific gateway (e.g. `https://eu-de.watson-orchestrate.cloud.ibm.com`).
- **`crn`**: The Cloud Resource Name identifying your service instance resource.
- **`agentId`**: The specific ID of your `dr_house` agent.
- **`agentEnvironmentId`**: The target runtime environment ID.

---

## Step 2 — Place Credentials in the Web App Code

The React component managing the consultation section is located at [`labs/ppth-web/src/components/Consult.tsx`](ppth-web/src/components/Consult.tsx).

1. Open [`labs/ppth-web/src/components/Consult.tsx`](ppth-web/src/components/Consult.tsx) in your editor.
2. Scroll to the **`useEffect`** hook starting around line 57. You will find several variables with `PLACE HERE` comments.
3. Replace the placeholder comment strings with your actual credentials.

```typescript
    const orchestrationID = "";  // PLACE HERE: orchestrationID 
    const hostURL = "";  // PLACE HERE: hostURL
    const crn = "";  // PLACE HERE: crn
    const agentId = "";  // PLACE HERE: agentId
    const agentEnvironmentId = "";  // PLACE HERE: agentEnvironmentId

```



> **How the loading works:**
> To keep the layout clean and modern, the web app renders a sleek, animated heartbeat monitor scanner (ECG) as a loading state inside the chat container (`#root`). 
>
> The moment the third-party script `wxoLoader.js` successfully initializes and injects the actual chat iframe, a custom CSS selector in [`labs/ppth-web/src/app/globals.css`](ppth-web/src/app/globals.css) (`.chat-frame:has(#root > *)`) automatically hides the loading placeholder and reveals the live chat!

---

## Step 3 — Run the Application Locally

Now that the credentials are wired up, it is time to start the Next.js development server.

1. Open your terminal.
2. Navigate to the project folder and start the local development server:

```bash
cd labs/ppth-web
npm install        # (Only needed once to install packages)
npm run dev
```

3. Open your browser and navigate to **[http://localhost:3000](http://localhost:3000)**.
4. Scroll down to the bottom of the page to find the **Consult** section (*"Take a case to Dr House."*) or click Open a case in the navigation bar.

---

## Step 4 — Configure Allowed Domains (Crucial Security Step)

When you first open the web application, you might notice that the chat widget loads, but then throws a **401 Unauthorized** or **403 Forbidden** error in your browser console, or simply fails to start a session.

This is a security feature. To protect your agent, watsonx Orchestrate limits which domains are permitted to embed and talk to your agent.

### How to allow localhost:

1. Return to the watsonx Orchestrate console.
2. Go back to your Settings **`your initials`** -> **Settings** -> **Embed Security**.
3. Under the **Security** tab, turn off the toggle switch.
4. Under the**Allowed origins**, click **Add domain** and input:
   ```text
   http://localhost:3000
   ```
5. Save the configuration.
6. Refresh your browser at `http://localhost:3000`. 
7. Click any of the patient sample cards to copy a test prompt (e.g. *"Patient PPTH-001 had sudden aphasia then a seizure. What is causing it?"*), paste it into the chat box, and watch the multi-agent diagnostic process run!

> **Notes:**
> For the sake of simplicity and due to time constraints, authorisation and authentication are not covered in this session. However, for a production application, it is essential to configure both of these functions correctly.
> If you deploy this web app to Vercel or any other cloud provider, you must come back to this screen and add your production URL (e.g., `https://your-app.vercel.app`) to this allowlist as well.

---

## Troubleshooting

| Problem | Potential Cause | Fix |
|---|---|---|
| **Chat stays on loading state (ECG animation)** | Host URL or script source is incorrect, or network is blocked. | Verify that `hostURL` matches your tenant region, and check your browser console for script loading blockages. |
| **Console shows 401/403 Errors on session start** | Missing allowed domains. | Follow Step 4 to add `http://localhost:3000` to the Web chat's Security domains in wxO console. |
| **Port 3000 is already in use** | Another dev server is running on port 3000. | Start the server on an alternative port: `npm run dev -- -p 3111` and add `http://localhost:3111` to the allowed domains list. |
| **Next.js complains about module imports** | Outdated or missing node packages. | Delete the `node_modules` folder, run `npm install` again, and restart the dev server. |

---
