
files <- c('main_paper.Rmd','SI.Rmd')
print(files)
for (f in files) {rmarkdown::render(f)}