
### Code to configure Quarto project with use on GitHub ###


# Initialize project w/ Git (if not done on project creation)
usethis::use_git()

# Create new GH repo for newly created Quarto Website project (after making first commit)
usethis::use_github(private = TRUE, protocol = "https")

# Set license
usethis::use_ccby_license()

# Create README file
usethis::use_readme_md()

# Initialize gh-pages branch to host rendered website
usethis::use_github_pages()

# Add /.quarto/ and /_site/ to .gitignore file

# Publish Quarto Website to "gh-pages" branch
system("quarto publish gh-pages")


# Create folder for storing GH Action YAML files
dir.create(".github/workflows", recursive = TRUE)
