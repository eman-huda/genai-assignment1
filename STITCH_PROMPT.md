# Google Stitch design prompt

Paste the text below into Google Stitch (https://stitch.withgoogle.com), choose **Web**, and generate.
Then ask Stitch for each of the four screens listed at the end.

---

Design a minimal, calm web app called "Restore & Sketch" for a university generative AI project.
An evaluator uses it to upload an image, run a model, and inspect the result and system information.

Style: soft and slightly girly but professional. Lavender-tinted white background (#FBFAFF),
baby blue control panels (#DCEBFB), lavender accents (#D9CFF6), a deeper lavender for primary
buttons (#7462C0), and slate navy text (#283552) instead of black. Rounded font (Nunito).
Generous rounded corners on panels and pill-shaped buttons. No gradients, no heavy shadows,
no all-caps labels, plenty of white space.

Layout: a left sidebar with the app name and four workspaces, each with a small status dot
(filled baby blue = ready, lavender ring = coming soon) and a server status box at the bottom.
The main area has the workspace title and one plain sentence of description, then two columns:
a baby blue control panel on the left (about 350 px wide) and the results area on the right.

Screens:
1. Universal Restoration: control panel with image source switch (Sample / Upload), a grid of
   sample pet photos, corruption chips (Salt and pepper, Gaussian blur, Occlusion, No corruption),
   severity switch (Low, Medium, High, Random) with the exact setting shown below it, an optional
   random seed field, a wide "Restore image" button, and a collapsible "Model details" section.
   Results area: a large square before/after comparison slider (model input on the left, restored
   on the right), a switch for "Compare" or "All images" (clean target, model input, restored,
   error map), four small readouts (model inference time, total time, PSNR input to restored,
   SSIM input to restored) and a "Corruption settings used" list. Download buttons at the top.
2. Hard-Routed Restoration (coming soon): same layout, greyed out, with a lavender notice that
   the model is not deployed yet, and a placeholder card showing four classifier probability bars
   (Clean, Salt and pepper, Gaussian blur, Occlusion).
3. Soft Mixture-of-Experts Restoration (coming soon): like screen 2 but with four routing weight
   bars (Identity, Salt-and-pepper expert, Blur expert, Occlusion expert).
4. Face-to-Sketch Generator: lavender control panel with photo source switch (Upload / Webcam),
   three style cards (Style 1 "Light, thin outlines", Style 2 "Bold strokes, dense shading",
   Style 3 "Balanced shading and detail"), a checkbox "Also draw the other two styles", and a
   "Generate sketch" button. Results area: the original photo and the generated pencil sketch
   side by side as large squares, "Download sketch" button, a row of the same photo in all three
   styles, and readouts for inference time, total time, resolution and style used.
