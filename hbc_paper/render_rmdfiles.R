files <- c('hbc_paper.Rmd','hbc_paper_SI.Rmd')
print(files)
for (f in files) {rmarkdown::render(f)}