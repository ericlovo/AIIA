import { buildJob, JOB_RECIPES, jobTestDefinition } from '../src/console/jobHelpers.ts'

const repos = [{ id: 'quality-fixture', name: 'Synthetic quality fixture', branch: 'main', dirty: false }]
const jobs = JOB_RECIPES.map(recipe => {
  const agent = buildJob({ recipeId: recipe.id, name: recipe.name, repoId: repos[0].id, interval: '1440', cap: '1' }, repos)
  return { recipe_id: recipe.id, agent, assignment: jobTestDefinition(agent) }
})
process.stdout.write(JSON.stringify(jobs))
