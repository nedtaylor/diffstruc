# Configuration file for the Sphinx documentation builder.

# -- Project information
import datetime
import os
import sys
import subprocess
import shutil


def env_truthy(name):
    """Return True when an environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def run_ford(app):
    """Run FORD to generate Fortran API documentation"""
    if env_truthy("DIFFSTRUC_SKIP_FORD") or env_truthy("SKIP_FORD"):
        print(
            "Skipping FORD documentation generation because "
            "DIFFSTRUC_SKIP_FORD or SKIP_FORD is set"
        )
        return

    ford_dir = os.path.abspath(os.path.join(app.confdir, "..", ".."))
    ford_otuput = os.path.join(app.confdir, "_static", "ford")
    project_file = os.path.join(ford_dir, "ford.md")

    print(f"Running FORD with config: {project_file}")
    result = subprocess.run(["ford", project_file, "-o", ford_otuput], cwd=ford_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FORD output:\n{result.stdout}")
        print(f"FORD errors:\n{result.stderr}")
    else:
        print("FORD documentation generated successfully")

def setup(app):
    app.add_css_file('custom.css')
    app.connect("builder-inited", run_ford)

# -- Project information

project = 'diffstruc'
copyright = f'{datetime.date.today().year}, diffstruc-developers'

# -- General configuration

# Identify the branch of the documentation
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    git_branch = os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")
else:
    git_branch = "main"  # or get from git directly with subprocess

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.extlinks',
    'sphinx_copybutton'
]

extlinks = {
    'git': ('https://github.com/nedtaylor/diffstruc/blob/' + git_branch + '/%s', 'git: %s')
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

exclude_patterns = ['_build', '.DS_Store', 'build']

# -- Options for HTML output

html_theme = 'furo'

templates_path = ['_templates']

# Add path for static files (will include FORD output)
html_static_path = ['_static']
html_css_files = [
    "custom.css",
]

html_theme_options = {
    "top_of_page_buttons": [ "view", "edit" ],
    "source_repository": "https://github.com/nedtaylor/diffstruc/",
    "source_branch": git_branch,
    "source_directory": "docs/source/",
}


html_context = {
    "display_github": True,
    "github_repo": "diffstruc",
    "github_user": "nedtaylor",
    "github_version": git_branch,
    "conf_py_path": "docs/source/",
    "page_source_suffix": ".rst",
}

# -- Options for EPUB output
epub_show_urls = 'footnote'
