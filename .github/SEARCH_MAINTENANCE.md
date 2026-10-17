# Search and answer-system discoverability

The documentation site is `https://izzatum.github.io/laplace-hypopinn/`.
The repository remains the source of code and executed notebooks. Clear research
content, correct citations and reachable HTML support discovery; neither metadata
nor this workflow guarantees indexing, AI citations or a particular search rank.

## GitHub settings

In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
The `documentation` workflow builds and validates on pull requests and publishes
from `main` when Pages is enabled. If setup is missing, the build remains useful
and deployment is skipped with a notice. After enabling Pages, run the workflow
manually from the Actions tab.

Set the About description to:

```text
Laplace HypoPINN: PyTorch physics-informed neural networks for earthquake hypocenter localization and uncertainty quantification. Flexible PINNs, KFAC, EFB, INF, VaLLA, reproducible experiments and eight tutorials.
```

Topics (semicolon separated for copying):

```text
physics-informed-neural-networks; pinn; laplace-approximation; bayesian-deep-learning; uncertainty-quantification; earthquake-location; seismology; geophysics; inverse-problems; pytorch; hypocenter-localization; scientific-machine-learning; kfac; eikonal-equation
```

Once the site responds successfully, set the About website to the site URL.
An integration can have permission to push code without permission to change
About, topics or Pages settings. Apply those settings through the owner account
if the integration receives `403 Resource not accessible by integration`.

## Verify ownership and submit the sitemap

1. Add the published URL in [Google Search Console](https://search.google.com/search-console/about)
   as a **URL-prefix property**, including `/laplace-hypopinn/`. The repository
   page on `github.com` is outside the site's verification scope.
2. Choose HTML-tag verification. Put only the supplied `content` value into the
   GitHub Actions repository variable `GOOGLE_SITE_VERIFICATION`. Rebuild/deploy,
   confirm the tag is present in the live home page, then complete verification
   in Search Console. Keep the value configured to preserve the verification.
3. Add the site in [Bing Webmaster Tools](https://www.bing.com/webmasters/about).
   Follow its current verification flow, or import the verified Search Console
   property when offered. For its HTML meta tag, set the repository variable
   `BING_SITE_VERIFICATION` to the supplied `content` value and redeploy.
4. Submit `https://izzatum.github.io/laplace-hypopinn/sitemap.xml` through each
   verified account. Inspect the home page and representative method/tutorial
   pages for crawling and index status. Request a recrawl after material updates.

Verification values are public ownership tags, not account passwords. The site
does not invent a token or emit empty verification tags. Webmaster accounts,
property verification and sitemap submissions are separate from a successful
GitHub Pages deployment.

After a successful deployment, the workflow verifies the published public
IndexNow key file, fetches the live sitemap and notifies the participating
engines using an explicit project-path `keyLocation`. The key is a public proof
of site control, not a password. The submission script rejects URLs outside this
project and can be inspected without network access using
`python scripts/submit_indexnow.py --dry-run`. A `200` or `202` response means the
notification was received; it does not mean a URL was indexed or ranked. Google
is not an IndexNow participant. A notification failure is reported separately
from a successful Pages deployment.

The site's project-path `llms.txt` is an optional source index for consumers that
choose to use it. Google explicitly says it ignores this file for visibility and
ranking. Human-readable FAQs help explain the research; no FAQ rich-result or AI
answer placement is promised. `SoftwareSourceCode` structured data describes
visible software facts and does not invent ratings or reviews.

## Maintain accurate crawlable content

```bash
python -m pip install -e '.[docs]'
python -m mkdocs build --strict
python scripts/check_docs.py site
python -m mkdocs serve
```

Keep canonical URLs, page titles/descriptions, citations, version and visible
content consistent. The build publishes the maintained documentation and
executed tutorial lessons; it excludes historical design notes and template
source. Notebook figures must retain meaningful descriptions and scientific
qualifications. Rebuilds must not manufacture a new content modification date.

Only `https://izzatum.github.io/robots.txt` can control robots rules for this
origin. A repository/project-path robots file cannot do that. Do not change
another repository to control origin-wide crawling. The site uses its actual
project path in canonical links, sitemap entries and any indexing notifications.

Track branded and research-intent queries separately: `Laplace HypoPINN`,
`physics informed earthquake localization`, `PINN hypocenter localization`, and
`Laplace approximation PyTorch PINN`. Use Search Console and Bing Webmaster Tools
to compare impressions, clicks, indexed URLs and available AI citation reports
over time. HTTP success is reachability, not evidence of index inclusion. Genuine
links from related papers and researcher websites can help readers discover the
project; do not fabricate mentions, citations or traffic.

## Primary references

- [Google SEO Starter Guide](https://developers.google.com/search/docs/fundamentals/seo-starter-guide)
- [Google guidance for generative AI search](https://developers.google.com/search/docs/fundamentals/ai-optimization-guide)
- [Google recrawl requests](https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl)
- [Bing AI Performance reporting](https://blogs.bing.com/webmaster/2026/2/Introducing-AI-Performance-in-Bing-Webmaster-Tools-Public-Preview/)
- [Bing IndexNow setup and verification](https://www.bing.com/indexnow/getstarted)
