# Prediction viewer

The viewer compares a text anchor, multimodal evidence and the corrected output.
It contains all 3,131 saved seed-3 validation observations: MOSI 229, MOSEI 1,871,
and IEMOCAP 1,031. The results tab separately reports the five-seed test summaries.
Correction sliders apply the readout equation to fixed saved components; they do
not run the model on new inputs. The interface is in English.

## Run locally

From the repository root, with Python 3.12 or later:

```bash
python start_demo.py
```

Open the address printed in the terminal on the same computer. A loopback address
such as `127.0.0.1` is local to that computer; it is not a public website address.
The viewer needs no model dependencies or GPU.

## Publish with GitHub Pages

1. Upload the repository to GitHub, including `.github/workflows/pages.yml`.
2. In **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.
3. In **Actions → Deploy prediction viewer**, choose **Run workflow** on the default branch.
4. After deployment succeeds, use the page URL shown in the workflow or **Settings → Pages**.

Only the six static viewer assets are published. No Python server, local transcript
overlay or model weights are required. Relative asset paths support project sites.
A deployment workflow does not mean a website is already live; the deployment must
complete in the owner's repository before sharing its URL.

See [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
