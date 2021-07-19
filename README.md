# Marp Pages

Marp Pages is a static site generator with Marp. You can:
- Make slides using Markdown syntax.
- Publish slide decks on GitHub Pages automatically with GitHub Actions.

## Usage

### Prerequisite

- [Node.js](https://nodejs.org/)
- Install dependency packages:
    ```bash
    npm i
    ```
- Python 3.6+

Make slide decks and put them in the `content/` directory. 
This work is based on [Marp](https://marp.app/), your slides should follow its settings and dialects.

### Preview

```bash
npm run preview
```

### Publish on GitHub Pages

1. Commit and push to your GitHub repository.
2. Setup GitHub Pages and change the branch to `gh-pages`:
    1. Navigate to **Settings/Pages**
    2. Select **Branch** to `gh-pages` and save.
