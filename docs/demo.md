# Harmony project website

Harmony: Bridging Audio Understanding and Editing through Agentic Collaboration.

This standalone static website contains the paper abstract and 12 audio comparisons, including Original, Harmony, AEC Agent, WavCraft original, WavCraft matched backbone, SmartDJ and MMEDIT. There is no dependency on the previous hosting service. All 84 WAV files preserve their original bytes.

## Publish with GitHub Pages

Push this directory as the root of a GitHub repository. In **Settings > Pages**, choose **Deploy from a branch**, then **main** and **/(root)**. The `.nojekyll` file keeps the existing HTML, JavaScript, CSS and audio files unchanged.

Use the URL shown by GitHub Pages after deployment. All asset links are relative, so project subpaths work without rewriting them.

Official instructions: https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site

## Preview locally

```bash
python -m http.server 8766 --bind 127.0.0.1
```

Open http://localhost:8766/ or forward port 8766 when using SSH.
