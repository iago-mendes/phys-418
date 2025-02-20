\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{paper}[Paper]
\LoadClass[
  aps,prd,amsmath,floats,floatfix,twocolumn,
  superscriptaddress,nofootinbib,showpacs
]{revtex4-2}

\usepackage[dvipsnames]{xcolor}
\usepackage{hyperref}
\hypersetup{
  colorlinks,
  citecolor={cyan!50!black},
  linkcolor={orange!60!black},
  urlcolor=violet,
}
\usepackage{orcidlink}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{mathtools}  % \coloneqq
\usepackage{tensor}
\usepackage{appendix}
\usepackage{bm} % boldface math
\usepackage{commath}  % differentials
\usepackage{physics}
\usepackage{bbm} % identity matrix
\usepackage{microtype}
\usepackage{enumitem}
\usepackage{siunitx}
\usepackage{csquotes}
\usepackage[capitalise]{cleveref}
\usepackage{pgf,tikz,pgfplots, tikz-3dplot}
\usepackage[labelfont=bf, width=.8\textwidth]{caption}
\usepackage{subcaption}

\renewcommand{\deg}{^\circ}

\newenvironment{sketch}[1][{}]{
  \begin{tikzpicture}
    \begin{axis}[
      width=6cm,
      height=6cm,
      axis lines = middle,
      x label style = {at={(current axis.right of origin)}, anchor = west},
      y label style = {at={(current axis.above origin)}, anchor = south},
      xtick = \empty,
      xtick style={draw=none},
      xticklabels=\empty,
      ytick = \empty,
      ytick style={draw=none},
      yticklabels=\empty
      samples=200,
      clip = false,
      #1
    ]
}{
    \end{axis}
  \end{tikzpicture}
}
