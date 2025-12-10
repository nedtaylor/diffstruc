project:
summary: A Fortran library for automatic differentiation
src_dir: ./src
output_dir: docs/html
preprocess: false
predocmark: !!
fpp_extensions: f90
display: public
         protected
         private
source: true
graph: true
search: true
md_extensions: markdown.extensions.toc
coloured_edges: true
sort: permission-alpha
author: Ned Thaddeus Taylor
print_creation_date: true
creation_date: %Y-%m-%d %H:%M %z
project_github: https://github.com/nedtaylor/diffstruc
project_download: https://github.com/nedtaylor/diffstruc/releases
github: https://github.com/nedtaylor
externalize: true

{!README.md!}
