# Compiling the main_paper.Rmd or SI.Rmd files
As of 3/12/2024 the compilation/rendering of the .Rmd --> .docx requires the 
`renv` library. 

## Starting the `renv` package

Begin by loading the existing renv environment: 
```
renv::load()

```
Don't be surprised if it takes a minute or more ... I'm still figuring out how to speed up the workflow with renv & rmarkdown etc.


Check to see if everything is okay with the latest snapshot:
```
renv:restore()

```


## Change in how the .Rmd file is knit

For whatever reason - the Rstudio knit button is painfully slow (and probably doesn't even knit the file...). 

The knit button is now replaced with a manual command entered in the R console:

```
rmarkdown::render('<name-of-your-file.Rmd>')
```






