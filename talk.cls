\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{talk}

% \LoadClass[10pt, aspectratio=43]{beamer}
\LoadClass[10pt, aspectratio=169]{beamer}

% PACKAGES
	\RequirePackage{multimedia}
	\RequirePackage[labelfont=bf, width=.9\textwidth]{caption}
	\RequirePackage{float}
	\RequirePackage{amsmath, amsfonts, amssymb, esint}
	\RequirePackage{pgf,tikz,pgfplots}
		\usetikzlibrary{arrows}
		\usepgfplotslibrary{polar}
		\pgfplotsset{compat=newest}
	\RequirePackage{array,ragged2e}
	\RequirePackage[most]{tcolorbox}
	\RequirePackage{multicol}
	\RequirePackage{newfloat}
	\RequirePackage{ifdraft}

% SETUP
	\usetheme[progressbar=frametitle]{metropolis}
	\usepackage{../beamercolorthemeomni}
	\usefonttheme{serif}
	\setbeamertemplate{section in toc}[sections numbered]

	\makeatletter
	\setlength{\metropolis@titleseparator@linewidth}{2pt}
	\setlength{\metropolis@progressonsectionpage@linewidth}{2pt}
	\setlength{\metropolis@progressinheadfoot@linewidth}{2pt}
	\makeatother

	\setbeamertemplate{footline}{
		\begin{beamercolorbox}[wd=\textwidth, sep=0.01\textheight, center]{footline}
			\insertshortauthor
			\hspace*{0.8\textwidth}
			\insertframenumber
		\end{beamercolorbox}
	}


% COMMANDS & ENVIRONMENTS
	% times
	\renewcommand{\t} {
		\cdot
	}

	% white box
	\newenvironment{whitebox}[1][{}] {
		\begin{tcolorbox}[colback=white, boxrule=0pt, frame hidden, boxsep=0pt, left=0pt, right=0pt, top=1pt, bottom=1pt, #1]
			\begin{center}
	} {
			\end{center}
		\end{tcolorbox}
	}

	% To-do note
	\newcommand{\red}[1]{\textcolor{red}{#1}}
	\newcommand{\todo}[1]{\red{\{#1\}}}

	% video listing
	\DeclareFloatingEnvironment[
		listname = {List of Videos},
		name = Video
	]{video}

	% video placeholder
	% optional argument = dimensions
	% required argument = placeholder content
	\newcommand{\videoplaceholder}[2][width = 0.85\textwidth, height = 0.478125\textwidth]{
		\begin{tcolorbox}[#1]
			\begin{center}
				#2
			\end{center}
		\end{tcolorbox}
	}

	% Only show heavy content if in final mode
	\newcommand{\iffinal}[1]{
		\ifdraft{%
			\boxed{\text{Content missing because of draft mode!}}%
		} {%
			#1%
		}%
	}

	% matrix
	\newcommand{\mtx}[1]{
		\begin{pmatrix}
			#1
		\end{pmatrix}
	}

	% determinant matrix
	\newcommand{\detmtx}[1]{
		\begin{vmatrix}
			#1
		\end{vmatrix}
	}

	% and
	\renewcommand{\and}{
		\quad\text{and}\quad
	}

	% then
	\newcommand{\then}{
		\quad\Rightarrow\quad
	}

	% partial derivative
	\newcommand{\pd}[2] {
		\frac{\partial #1}{\partial #2}
	}
