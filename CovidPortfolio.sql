Select *
FROM PortfolioProject..CovidDeaths
Where continent is not null
ORDER BY 3, 4


--Select Data that we are going to be using
Select location, date, total_cases, new_cases, total_deaths, population
FROM PortfolioProject..CovidDeaths
ORDER BY 1,2 

--Looking at Total Cases vs Total Deaths
--Shows likelihood of dying per country
Select location, date, total_cases, total_deaths, (total_deaths/total_cases) * 100 as
Death_Percentage
FROM PortfolioProject..CovidDeaths
WHERE location like '%states%'
ORDER BY 1,2 

--Looking at Total Cases vs Population
--Shows what percentage of pop got Covid
Select location, date,population, total_cases, (total_cases/population) * 100 as 
Death_Percentage
From PortfolioProject..CovidDeaths
--WHERE location like '%states%'
Order By 1,2 

--Looking at Countries with highest infection rate compared to Population 
Select location, population, MAX(total_cases) as HighestInfectionCount, MAX((total_cases/population)) * 100 as 
 PercentPopulationInfected
From PortfolioProject..CovidDeaths
--WHERE location like '%states%'
Where continent is not null
Group By population, location
Order By  PercentPopulationInfected Desc

--Showing countries with highest death count per population 
Select location, MAX(cast(Total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths
Where continent is not null
--WHERE location like '%states%'
Group By location
Order By  TotalDeathCount Desc

--Break down by continent
--Select location, MAX(cast(Total_deaths as int)) as TotalDeathCount
--From PortfolioProject..CovidDeaths
--Where continent is null
----WHERE location like '%states%'
--Group By location
--Order By TotalDeathCount Desc

--Break down by continent
--Showing continents with highest death count 
Select continent, MAX(cast(Total_deaths as int)) as TotalDeathCount
From PortfolioProject..CovidDeaths
Where continent is not null
--WHERE location like '%states%'
Group By continent
Order By TotalDeathCount Desc


--Global numbers
Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, SUM(cast(new_deaths as int))
/SUM(new_cases) * 100 as Death_Percentage
FROM PortfolioProject..CovidDeaths
WHERE continent is not null
--Group By date 
ORDER BY 1,2 

--Join tables. Look at total popultaion vs vaccinations
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
,SUM(CONVERT(bigint, vac.new_vaccinations)) OVER(Partition by dea.location Order by dea.location, 
 CONVERT(date, dea.date)) as RollingPeopleVaccinated
 --(RollingPeopleVaccinated/popultaion) *100
From PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
    On dea.location = vac.location
    and dea.date = vac.date
Where dea.continent is not null
Order by 2,3

--Use CTE

With PopvsVacc(Continent, location, data, popultaion,new_vaccinations, RollingPeopleVaccinated)
as 
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
,SUM(CONVERT(bigint, vac.new_vaccinations)) OVER(Partition by dea.location Order by dea.location,
 CONVERT(date, dea.date)) as RollingPeopleVaccinated
 --(RollingPeopleVaccinated/popultaion) *100
From PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
    On dea.location = vac.location
    and dea.date = vac.date
Where dea.continent is not null)
Select *, (RollingPeopleVaccinated/population) *100
From PopvsVacc


--Temp Table
Drop Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
location nvarchar(255),
Date datetime,
population numeric,
New_vaccinations numeric,
RollingPeopleVaccinated numeric
)
Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
,SUM(CONVERT(bigint, vac.new_vaccinations)) OVER(Partition by dea.location Order by dea.location,
 CONVERT(date, dea.date)) as RollingPeopleVaccinated
 --(RollingPeopleVaccinated/popultaion) *100
From PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
    On dea.location = vac.location
    and dea.date = vac.date
Where dea.continent is not null

Select *, (RollingPeopleVaccinated/population) *100
From #PercentPopulationVaccinated

--Creating view to store data for later visualizations
Create view PercentPopulationVaccinated as 
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
,SUM(CONVERT(bigint, vac.new_vaccinations)) OVER(Partition by dea.location Order by dea.location,
 CONVERT(date, dea.date)) as RollingPeopleVaccinated
 --(RollingPeopleVaccinated/popultaion) *100
From PortfolioProject..CovidDeaths dea
Join PortfolioProject..CovidVaccinations vac
    On dea.location = vac.location
    and dea.date = vac.date
Where dea.continent is not null

Select * 
From PercentPopulationVaccinated
